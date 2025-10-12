from collections import defaultdict
from typing import Dict

from bioc import BioCDocument
from collections import namedtuple

StoredEntity = namedtuple('StoredEntity', ['text', 'canonical_name', 'biolink_type'])
StoredRelationship = namedtuple('StoredRelationship', ['subject', 'predicate', 'object'])


def parse_annotations(passages) -> defaultdict[list]:
    """
    Parses the annotations from a bioc dataset
    :param passages:
    :return:
    """
    parsed = defaultdict(list)
    for passage in passages:
        for annotation in passage.annotations:
            # ontology_ref = om.validate_and_enrich_entity(annotation.text)
            canonical_name = annotation.text
            parsed[annotation.infons['identifier']].append(stored_relationship.StoredEntity(text=annotation.text, canonical_name=canonical_name, biolink_type=annotation.infons['type']))
    return parsed

def parse_relationships(relations: list, annotations) -> list:
    """
    Parses the relationships from a bioc dataset
    :param relations:
    :param annotations:
    :return:
    """
    processed = []
    for relation in relations:
        processed += [namedtuple(subject=annotations.get(relation.infons['entity1'], [relation.infons['entity1']])[0], predicate=relation.infons['type'], object=annotations.get(relation.infons['entity2'], [relation.infons['entity2']])[0], confidence=1, evidence='')]
    return processed


def parse_document(document: BioCDocument) -> Dict:
    """
    parses a bioc document
    :param document: the document in bioc format
    :return: the dict of relations, text, annotations
    """
    annotations = parse_annotations(document.passages)

    return {
        'relations': parse_relationships(document.relations, annotations),
        'text': '/n'.join([passage.text for passage in document.passages]),
        'annotations': annotations
    }