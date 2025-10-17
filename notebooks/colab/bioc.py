import bioc
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import torch
from bioc import biocxml
from torch.utils.data import Dataset
from transformers import AutoTokenizer

ENTITY_TOKENS = [
            "[E1]", "[/E1]",
            "[E2]", "[/E2]"
]


class BioCDataset(Dataset):
    """
    Dataset for relation extraction
    """

    def __init__(self, bioc_file_path: str, config, tokenizer=None):
        self.config = config
        self.data = self._load_and_process_bioc_data(bioc_file_path)
        if not tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, additional_special_tokens=ENTITY_TOKENS)
        self.tokenizer = tokenizer

    def parse_annotations(self, passages) -> defaultdict:
        """
        Parsing of annotations from bioc format
        """
        parsed = defaultdict(list)
        cumulative_offset = 0

        for passage in passages:
            passage_text = passage.text
            passage_offset = passage.offset or 0

            for annotation in passage.annotations:
                # Calculate position in full document text
                annotation_start = annotation.locations[0].offset - passage_offset + cumulative_offset
                annotation_end = annotation.locations[0].end - passage_offset + cumulative_offset

                # Store as dict with position info
                entity_data = {
                    'text': annotation.text,
                    'canonical_name': annotation.text,  # You can add ontology enrichment here
                    'biolink_type': annotation.infons.get('type', 'ENTITY'),
                    'start_pos': annotation_start,
                    'end_pos': annotation_end,
                    'identifier': annotation.infons['identifier']
                }

                parsed[annotation.infons['identifier']].append(entity_data)

            # Update cumulative offset for next passage (including newline)
            cumulative_offset += len(passage_text) + 1  # +1 for '\n'

        return parsed

    def parse_relationships(self, relations: list, annotations: defaultdict) -> List[Dict]:
        """
        parses relationships from bioc
        """
        processed = []

        for relation in relations:
            entity1_id = relation.infons.get('entity1')
            entity2_id = relation.infons.get('entity2')
            relation_type = relation.infons.get('type', 'related_to')

            # Get entities by ID (take first if multiple)
            subject_entities = annotations.get(entity1_id, [])
            object_entities = annotations.get(entity2_id, [])

            if subject_entities and object_entities:
                subject = subject_entities[0]
                obj = object_entities[0]

                relationship_data = {
                    'subject': subject,
                    'predicate': relation_type,
                    'object': obj,
                    'confidence': 1.0,
                    'evidence': ''
                }
                processed.append(relationship_data)

        return processed

    def parse_document(self, document) -> Dict:
        """
        """
        annotations = self.parse_annotations(document.passages)
        full_text = '\n'.join([passage.text for passage in document.passages])

        return {
            'relations': self.parse_relationships(document.relations, annotations),
            'text': full_text,
            'annotations': annotations,
            'document_id': document.id
        }

    def _load_and_process_bioc_data(self, file_path: str) -> List[Dict]:
        """Load and process BioC data """
        with open(file_path, 'r', encoding='utf-8') as f:
            collection = bioc.load(f)

        processed_examples = []

        for document in collection.documents:

            parsed_doc = self.parse_document(document)

            # Convert each relationship to a training example
            for relation in parsed_doc['relations']:
                example = {
                    'text': parsed_doc['text'],
                    'subject': relation['subject'],
                    'object': relation['object'],
                    'predicate': relation['predicate'],
                    'confidence': relation['confidence'],
                    'document_id': parsed_doc['document_id']
                }
                processed_examples.append(example)

        print(f"Loaded {len(processed_examples)} relation examples from {len(collection.documents)} documents")
        return processed_examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract components
        text = item['text']
        subject = item['subject']
        obj = item['object']
        predicate = item['predicate']

        # Create entity dictionaries for processing
        entity1 = {
            'text': subject['text'],
            'start': subject['start_pos'],
            'end': subject['end_pos'],
            'type': subject['biolink_type']
        }

        entity2 = {
            'text': obj['text'],
            'start': obj['start_pos'],
            'end': obj['end_pos'],
            'type': obj['biolink_type']
        }

        # Create marked text with entity markers
        marked_text = self._create_marked_text(text, entity1, entity2)

        # Tokenize
        encoding = self.tokenizer(
            marked_text,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt"
        )

        # Find entity positions in tokenized text
        entity1_pos, entity2_pos = self._find_entity_positions(encoding, entity1, entity2)

        # Convert predicate to label
        relation_label = self.config.relation_to_id.get(predicate, 0)

        # Return tensors
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "entity1_pos": torch.tensor(entity1_pos, dtype=torch.long),
            "entity2_pos": torch.tensor(entity2_pos, dtype=torch.long),
            "relation_label": torch.tensor(relation_label, dtype=torch.long),
            "entity1_type": entity1['type'],
            "entity2_type": entity2['type'],
            "document_id": item['document_id'],
        }

    def _create_marked_text(self, text: str, entity1: Dict, entity2: Dict) -> str:
        """Mark entities in text with special tokens"""
        # Sort entities by position (reverse order for safe insertion)
        entities = [
            (entity1, "[E1]", "[/E1]"),
            (entity2, "[E2]", "[/E2]")
        ]
        entities.sort(key=lambda x: x[0]["start"], reverse=True)

        marked_text = text
        for entity, start_marker, end_marker in entities:
            start = entity["start"]
            end = entity["end"]

            # Validate positions
            if 0 <= start < end <= len(text):
                marked_text = (marked_text[:start] + start_marker +
                               marked_text[start:end] + end_marker +
                               marked_text[end:])
            else:
                # Handle invalid positions - just add markers at entity text
                entity_text = entity["text"]
                if entity_text in marked_text:
                    marked_text = marked_text.replace(
                        entity_text,
                        f"{start_marker}{entity_text}{end_marker}",
                        1  # Replace only first occurrence
                    )

        return marked_text

    def _find_entity_positions(self, encoding, entity1: Dict, entity2: Dict) -> Tuple[int, int]:
        """Find positions of entity markers in tokenized text"""
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze())

        e1_pos = 0
        e2_pos = 0

        # Look for entity markers
        for i, token in enumerate(tokens):
            if "[E1]" in token or token == "[E1]":
                e1_pos = min(i + 1, len(tokens) - 1)  # Position after start marker
            elif "[E2]" in token or token == "[E2]":
                e2_pos = min(i + 1, len(tokens) - 1)  # Position after start marker

        return e1_pos, e2_pos

# BioCPreprocessor
class BioCPreprocessor:
    """Data analysis and preprocessing using your parsing functions"""

    def __init__(self, file):
        with open(file, 'r') as bio_red_file:
            self.collection = biocxml.load(bio_red_file)

    def parse_annotations(self, passages) -> defaultdict:
        """parse annotations"""
        parsed = defaultdict(list)
        for passage in passages:
            for annotation in passage.annotations:
                canonical_name = annotation.text
                parsed[annotation.infons['identifier']].append({
                    'text': annotation.text,
                    'canonical_name': canonical_name,
                    'biolink_type': annotation.infons.get('type', 'ENTITY')
                })
        return parsed

    def parse_relationships(self, relations: list, annotations: defaultdict) -> List[Dict]:
        """parse relationships"""
        processed = []
        for relation in relations:
            entity1_id = relation.infons.get('entity1')
            entity2_id = relation.infons.get('entity2')

            subject_entities = annotations.get(entity1_id, [])
            object_entities = annotations.get(entity2_id, [])

            if subject_entities and object_entities:
                processed.append({
                    'subject': subject_entities[0],
                    'predicate': relation.infons.get('type', 'related_to'),
                    'object': object_entities[0],
                    'confidence': 1.0,
                    'evidence': ''
                })
        return processed

    def parse_document(self, document) -> Dict:
        """parse a document"""
        annotations = self.parse_annotations(document.passages)
        return {
            'relations': self.parse_relationships(document.relations, annotations),
            'text': '\n'.join([passage.text for passage in document.passages]),
            'annotations': annotations
        }

    def analyze_bioc_file(self) -> Dict:
        """Analyze BioC file"""

        stats = {
            'num_documents': len(self.collection.documents),
            'relation_types': defaultdict(int),
            'entity_types': defaultdict(int),
            'num_relations': 0,
            'num_entities': 0
        }

        for document in self.collection.documents:
            parsed_doc = self.parse_document(document)

            # Count stats
            stats['num_relations'] += len(parsed_doc['relations'])

            for relation in parsed_doc['relations']:
                stats['relation_types'][relation['predicate']] += 1

            for entity_list in parsed_doc['annotations'].values():
                stats['num_entities'] += len(entity_list)
                for entity in entity_list:
                    stats['entity_types'][entity['biolink_type']] += 1

        return stats