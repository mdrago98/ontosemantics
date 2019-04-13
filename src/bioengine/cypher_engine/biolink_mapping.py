from collections import OrderedDict

from py2neo import Relationship, Node

from cypher_engine.models import AnatomicalEntityToAnatomicalEntityAssociation, \
    ChemicalToDiseaseOrPhenotypicFeatureAssociation, ChemicalToGeneAssociation, ChemicalToThingAssociation, \
    GeneToGeneAssociation, GeneToThingAssociation, ThingToDiseaseOrPhenotypicFeatureAssociation, \
    DiseaseOrPhenotypicFeatureAssociationToThingAssociation, MolecularEntity, OrganismalEntity, \
    DiseaseOrPhenotypicFeature, ChemicalSubstance, Class, NamedThing, Association
from preprocessor.extensions.noun_verb_noun import Relation

associations = {
    AnatomicalEntityToAnatomicalEntityAssociation: (['FMA', 'ZEBRAFISH_ANATOMICAL_ONTOLOGY'],
                                                    ['FMA', 'ZEBRAFISH_ANATOMICAL_ONTOLOGY']),
    ChemicalToDiseaseOrPhenotypicFeatureAssociation: (['CHEBI'], ['DOID', 'HUMAN_PHENOTYPE']),
    ChemicalToGeneAssociation: (['CHEBI'], ['GO']),
    ChemicalToThingAssociation: (['CHEBI'], ['*']),
    GeneToGeneAssociation: (['GO'], ['GO']),
    GeneToThingAssociation: (['GO'], ['*']),
    ThingToDiseaseOrPhenotypicFeatureAssociation: (['*'], ['DOID', 'HUMAN_PHENOTYPE']),
    DiseaseOrPhenotypicFeatureAssociationToThingAssociation: (['DOID', 'HUMAN_PHENOTYPE'], ['*'])
}

term_mapping = {
    MolecularEntity: ['GO'],
    OrganismalEntity: ['FMA', 'ZEBRAFISH_ANATOMICAL_ONTOLOGY', 'UBERON', 'CL'],
    DiseaseOrPhenotypicFeature: ['DOID', 'HUMAN_PHENOTYPE', 'CMPO'],
    ChemicalSubstance: ['CHEBI']
}


def get_association(relation: Relation, terms: dict) -> dict:
    """
    A method that returns the appropriate association term
    :param terms: a dictionary of enriched terms from the ontology store
    :param relation: the relation
    """
    subject_terms = terms[relation.effector]
    object_terms = terms[list(filter(lambda x: isinstance(x, str), relation.effectee))[0]]
    subject_query = '*' if len(subject_terms) > 1 else subject_terms[0].ontology_prefix
    subject_query = '*' if len(list(filter(lambda x: subject_query in x[1][0], associations.items()))) is 0 \
        else subject_query
    object_query = '*' if len(object_terms) > 1 else object_terms[0].ontology_prefix
    object_query = '*' if len(list(filter(lambda x: object_query in x[1][0], associations.items()))) is 0 \
        else object_query
    return {association: requirements for association, requirements in associations.items()
            if subject_query in requirements[0]
            and object_query in requirements[1]}


def get_mapping(term: Class) -> OrderedDict:
    """
    A method for returning the biolink mapping from the source object
    :param term: A neo4j OGM class
    :return: a dict of possible mappings
    """
    mapping = {biolink_entity: requirements for biolink_entity, requirements in term_mapping.items()
               if term.ontology_prefix in requirements}
    if len(mapping) is 0:
        mapping = {NamedThing: ['*']}
    return OrderedDict(sorted(mapping.items(), key=lambda kv: (-len(kv[1]))))


def get_biolink_object(token: str, terms):
    term = terms[token]
    biolink_mapping = list(get_mapping(term[0]).items())[0][0]
    ogm = terms[token][0]
    return biolink_mapping(id=ogm.iri,
                           name=ogm.label,
                           category=[term.ontology_iri for term in terms[token]],
                           iri=ogm.iri,
                           full_name=ogm.label,
                           synonym=ogm.synonym if hasattr(ogm, 'synonym') else [],
                           description=ogm.description)


def get_nodes_from_biolink_object(biolink_named_thing: NamedThing):
    """
    A function that returns a neo4j node from a biolink named thing or subclass of
    :param biolink_named_thing: the biolink named thing
    :return: a neo4j node
    """
    return Node(biolink_named_thing.__class__.__name__, **biolink_named_thing.__dict__)


def get_relationship_from_biolink(biolink_subject: NamedThing,
                                  biolink_association: Association,
                                  biolink_object: NamedThing):
    """
    A function that returns the neo4j association from a biolink association
    :param biolink_subject: the biolink representation of the subject or effector
    :param biolink_association: the biolink representation of the association between the subject and object
    :param biolink_object: the biolink representation of the biolink object
    :return: a neo4j relationship
    """
    return Relationship(biolink_subject, biolink_association.__class__.__name__,
                        biolink_object, **biolink_association.__dict__)
