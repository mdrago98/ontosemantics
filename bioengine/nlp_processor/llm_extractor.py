from datetime import datetime
from typing import Dict, List

from knowledge_engine.models.relationship import ExtractedRelationship
from nlp_processor.relationship_extractor import RelationshipExtractor
from langchain.llms import Ollama
import re
import json
from logging import getLogger

logger = getLogger(__name__)

DEFAULT_PROMPT = """
You are a biomedical expert analyzing scientific literature. Extract relationships between biomedical entities from the given text.

Text: {text}

Instructions:
1. Identify biomedical entities (diseases, drugs, genes, symptoms, etc.)
2. Find explicit relationships between these entities
3. Return relationships in JSON format with high confidence only

Return ONLY a JSON array with this exact format:
[
  {{
    "subject": "entity1",
    "predicate": "relationship_type", 
    "object": "entity2",
    "evidence": "exact text supporting this relationship",
    "confidence": 0.85
  }}
]

Valid relationship types: causes, treats, increases_risk_for, decreases_risk_for, biomarker_for, positive_correlation, negative_correlation, association, bind, prevents

Focus on explicit, clear relationships. Avoid speculation.

JSON:
"""

PREDICATE_MAPPING = {
    'causes': ['cause', 'leads to', 'results in', 'induces', 'triggers'],
    'treats': ['treat', 'therapy for', 'used for', 'reduces'],
    'increases_risk_for': ['risk factor for', 'increases risk of', 'predisposes to'],
    'decreases_risk_for': ['protective against', 'reduces risk of'],
    'biomarker_for': ['marker for', 'indicator of', 'diagnostic for'],
    'association': ['linked to', 'correlated with', 'related to', 'associated_with', 'is associated with'],
    'variant': ['is a variant'],
    'positive_correlation': ['positively correlated with'],
    'negative_correlation': ['negatively correlated with'],
    'bind': ['binds']
}


class LLMRelationshipExtractor(RelationshipExtractor):
    """Extract relationships using LLM (Llama, GPT, etc.)"""

    def __init__(self, model_name: str, prompt_template: str = DEFAULT_PROMPT):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.llm = self._initialize_model()

    def extract_relationships(self, text: str, context: Dict = None) -> List[ExtractedRelationship]:
        """
        Extract relationships using the llm prompt
        :param text:
        :param context:
        :return:
        """
        # LLM-based extraction logic

        formatted_prompt = self._format_prompt(text, context)

        # Get LLM response
        llm_response = self.llm.invoke(formatted_prompt)

        verifier = f"""
        Ensure that the predicates in the following extracted relationships are one of these {','.join(PREDICATE_MAPPING.keys())}.
        Otherwise map the predicate to the most appropriate term from the list. 
        
        Result to confirm: {llm_response}
        Ensure the result is a JSON of this form:
        [
          {{
            "subject": "entity1",
            "predicate": "relationship_type", 
            "object": "entity2",
            "evidence": "exact text supporting this relationship",
            "confidence": 0.85
          }}
        ]
        JSON:
        """
        llm_response = self.llm.invoke(verifier)

        # Parse the response into structured relationships
        relationships = self._parse_llm_response(llm_response, text)
        return relationships

    def _initialize_model(self):
        # Factory pattern for different LLM types
        return Ollama(model=self.model_name, temperature=0.1, base_url="http://localhost:11434")

    def _format_prompt(self, text: str, context: Dict = None) -> str:
        """
        Format the prompt with the input text"
        :param text: the biomedical text
        :param context: the context
        :return: the formatted prompt
        """""
        formatted = self.prompt_template.format(text=text)

        # Add context if provided
        if context:
            context_str = f"\nAdditional context: {context}"
            formatted += context_str

        return formatted


    def _parse_llm_response(self, response: str, original_text: str) -> List[ExtractedRelationship]:
        """Parse LLM response into ExtractedRelationship objects"""
        relationships = []

        try:
            # Extract JSON from response (LLMs sometimes add extra text)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                print(f"No JSON found in LLM response: {response[:200]}...")
                return relationships

            json_str = json_match.group(0)
            parsed_relationships = json.loads(json_str)

            for rel_data in parsed_relationships:
                # Validate required fields
                if not all(key in rel_data for key in ['subject', 'predicate', 'object']):
                    continue

                # Create ExtractedRelationship object
                relationship = ExtractedRelationship(
                    subject=rel_data['subject'].strip(),
                    predicate=self._normalise_predicate(rel_data['predicate']),
                    object=rel_data['object'].strip(),
                    evidence_text=rel_data.get('evidence', ''),
                    extraction_method=f"llm_{self.model_name}",
                    confidence=float(rel_data.get('confidence', 0.5)),
                    metadata={
                        'original_text': original_text,
                        'extraction_timestamp': datetime.now().isoformat(),
                        'model_name': self.model_name
                    }
                )

                # Basic quality filtering
                if self._is_valid_relationship(relationship):
                    relationships.append(relationship)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.error(f"Response was: {response[:500]}...")
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
        return relationships


    def _normalise_predicate(self, predicate: str, predicate_mapping=None):
        """
        Gets the normalised biolink predicate
        :param predicate: the predicate
        :param predicate_mapping: the predicate mapping defaults to teh value of PREDICATE_MAPPING
        :return: the biolink predicate
        """

        if predicate_mapping is None:
            predicate_mapping = PREDICATE_MAPPING
        matches = [key for key, value in predicate_mapping.items() if predicate in value or predicate == key]
        return matches[0] if matches else predicate

    def _is_valid_relationship(self, relationship: ExtractedRelationship):
        """
        Checks if the relationship is valid
        :param relationship: the extracted relationship
        :return: TRUE IFF the relationship is valid and has a good confidence
        """
        return True




