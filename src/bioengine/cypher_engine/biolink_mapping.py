from collections import OrderedDict

from py2neo import Relationship, Node, Subgraph

from src.bioengine.cypher_engine.models import Class
from biolinkmodel.datamodel import AnatomicalEntityToAnatomicalEntityAssociation, \
    ChemicalToDiseaseOrPhenotypicFeatureAssociation, ChemicalToGeneAssociation, ChemicalToThingAssociation, \
    GeneToGeneAssociation, GeneToThingAssociation, ThingToDiseaseOrPhenotypicFeatureAssociation, \
    DiseaseOrPhenotypicFeatureAssociationToThingAssociation, MolecularEntity, OrganismalEntity, \
    ChemicalSubstance, NamedThing, Association, Disease, PhenotypicFeature, Cell, \
    CellularComponent, BiologicalProcess, MolecularActivity
from settings import Config
from src.bioengine.utils.pythonic_name import get_pythonic_name

associations = {
    AnatomicalEntityToAnatomicalEntityAssociation: (['OrganismalEntity', 'Cell', 'CellularComponent'],
                                                    ['OrganismalEntity', 'Cell', 'CellularComponent']),
    ChemicalToDiseaseOrPhenotypicFeatureAssociation: (['ChemicalSubstance'], ['Disease', 'PhenotypicFeature']),
    ChemicalToGeneAssociation: (['ChemicalSubstance'], ['MolecularEntity']),
    ChemicalToThingAssociation: (['ChemicalSubstance'], ['NamedThing']),
    GeneToGeneAssociation: (['MolecularEntity'], ['MolecularEntity']),
    GeneToThingAssociation: (['MolecularEntity'], ['NamedThing']),
    ThingToDiseaseOrPhenotypicFeatureAssociation: (['NamedThing'], ['Disease', 'PhenotypicFeature']),
    DiseaseOrPhenotypicFeatureAssociationToThingAssociation: (['Disease', 'PhenotypicFeature'], ['NamedThing'])
}

term_mapping = {
    MolecularEntity: ['GO'],
    OrganismalEntity: ['FMA', 'ZEBRAFISH_ANATOMICAL_ONTOLOGY', 'UBERON'],
    Cell: ['CL'],
    CellularComponent: ['CellularComponent'],
    BiologicalProcess: ['BiologicalProcess'],
    MolecularActivity: ['MolecularFunction'],
    Disease: ['DOID'],
    PhenotypicFeature: ['HUMAN_PHENOTYPE', 'CMPO'],
    ChemicalSubstance: ['CHEBI']
}


def get_mapping(term: Class) -> OrderedDict:
    """
    A method for returning the biolink mapping from the source object
    :param term: A neo4j OGM class
    :return: a dict of possible mappings
    """
    mapping = {biolink_entity: requirements for biolink_entity, requirements in term_mapping.items()
               if term.ontology_prefix in requirements or (hasattr(term, 'annotation_has_obo_namespace')
                                                           and term.annotation_has_obo_namespace in requirements)}
    if len(mapping) is 0:
        mapping = {NamedThing: ['*']}
    return OrderedDict(sorted(mapping.items(), key=lambda kv: (-len(kv[1]))))


def get_relationship_node(token: str, terms: dict) -> tuple:
    """
    A function for getting a knowledge graph node for a given token if it exists in the term dictionary
    :param token: the token to represent
    :param terms: a dictionary of terms: ontology_nodes
    :return: a tuple representing knowledge graph node and a list of subgraph relationships
    """
    main = None
    sub_relationships = None
    if token in terms:
        main = get_nodes_from_biolink_object(get_biolink_object(token, terms))
        sub_nodes = terms[token]
        for alternate_map in sub_nodes:
            if alternate_map.iri != main['iri']:
                alternate_node = Node('OntologyClass',
                                      label=alternate_map.label,
                                      iri=alternate_map.iri,
                                      ontology_iri=alternate_map.ontology_iri,
                                      ontology_name=alternate_map.ontology_name)
                entity_relation = Relationship(main, 'could_refer_to', alternate_node)
                sub_relationships = entity_relation if sub_relationships is None\
                    else sub_relationships | entity_relation
    return main, sub_relationships


def get_providers(providers: list, provider_type: str = 'Author') -> list:
    """
    A method that maps provider types from biolink to neo4j property node
    :param sub_graph: a neo4j subgraph instance
    :param providers: a list of provider names or groups
    :param provider_type: the type of provider (author, publisher etc.
    :return: a list of neo4j nodes
    """
    return [Node('Provider', name=provider, type=provider_type) for provider in providers]


def get_publication_node(abstract: str, pmid: str, name: str, conf: Config = None) -> Node:
    """
    A method that creates a Publication node
    :param name: Name of publication
    :param conf: config object
    :param abstract: the abstract of the publication
    :param pmid: the pmid of the publication
    :return: the publication node
    """
    if conf is None:
        conf = Config()
    iri = f'{conf.get_property("pubmed_base_iri")}/?term={pmid}[uid]'
    return Node('Publication', abstract=abstract, pmid=pmid, name=name, iri=iri)


def link_publication_to_provider(providers: list, publication: Node, sub_graph: Subgraph) -> Subgraph:
    """
    A method that generates relationships between the providers and  publications
    :param sub_graph: A neo4j sub_graph
    :param providers: a list of provider nodes
    :param publication: a publication node
    :return: a subgraph of relationships between providers and publication
    """
    for provider in providers:
        relation = Relationship(provider, 'provided_by', publication)
        sub_graph = provider if sub_graph is not None else sub_graph | provider
        sub_graph |= relation
    return sub_graph


def link_entities_to_publication(entities: list, publication: Node, sub_graph) -> Subgraph:
    """
    A method that generates relationships between entities and a publication
    :param sub_graph: a neo4j subgraph
    :param entities: a list of entity nodes
    :param publication: a publication node
    :return: a sub_graph relationships between entities and publications
    """
    for entity in entities:
        if 'OntologyClass' not in list(entity.labels):
            sub_graph |= Relationship(entity, 'located_in', publication)
    return sub_graph


def get_biolink_object(token: str, terms) -> NamedThing:
    """
    A function that obtains the biolink representation from a token
    :param token: the token to represent
    :param terms: a dictionary of terms: ontology nodes
    :return: a biolink named thing or subclass of named thing
    """
    mapping = None
    if token in terms:
        term = terms[token]
        biolink_mapping = list(get_mapping(term[0]).items())[0][0]
        ogm = terms[token][0]
        mapping = biolink_mapping(id=ogm.iri,
                                  name=ogm.label,
                                  category=[term.ontology_iri for term in terms[token]],
                                  iri=ogm.iri,
                                  full_name=ogm.label,
                                  synonym=ogm.synonym if hasattr(ogm, 'synonym') and ogm.synonym is not None else [],
                                  description=ogm.description)
    return mapping


def get_nodes_from_biolink_object(biolink_named_thing: NamedThing) -> Node:
    """
    A function that returns a neo4j node from a biolink named thing or subclass of
    :param biolink_named_thing: the biolink named thing
    :return: a neo4j node
    """
    node = None
    if biolink_named_thing is not None:
        node = Node(biolink_named_thing.__class__.__name__, **biolink_named_thing.__dict__)
    return node


def get_association(relation: str, subject_node: Node, object_node: Node, is_negated: bool,
                    association_config: dict = None):
    """
    A function that returns the most appropriate association for a subject/object pair
    :param relation: the name of the relation
    :param subject_node: the subject node
    :param object_node: the object node
    :param is_negated: a flag that indicates the ration is negated
    :param association_config: a dict representing the appropriate relation and conditions
    :return: an instantiated biolink association
    """
    if association_config is None:
        association_config = associations
    most_relevant_relation = list(get_biolink_association(subject_node, object_node, association_config).items())[0][0]
    biolink_relation = most_relevant_relation(id=0,
                                              subject=subject_node['id'],
                                              relation=relation,
                                              object=object_node['id'],
                                              negated=is_negated)
    return get_relationship_from_biolink(subject_node, biolink_relation, object_node)


def get_biolink_association(subject_node: Node, object_node: Node, association_config: dict = None) -> dict:
    """
    A method that returns the appropriate association term
    """
    if association_config is None:
        association_config = associations
    subject_query = list(subject_node.labels)[0]
    object_query = list(object_node.labels)[0]
    association = {association: requirements for association, requirements in association_config.items()
                   if subject_query in requirements[0]
                   and object_query in requirements[1]}
    if len(association) is 0:
        association = {Association: ['*']}
    return association


def get_relationship_from_biolink(biolink_subject: Node,
                                  biolink_association: Association,
                                  biolink_object: Node):
    """
    A function that returns the neo4j association from a biolink association
    :param biolink_subject: the biolink representation of the subject or effector
    :param biolink_association: the biolink representation of the association between the subject and object
    :param biolink_object: the biolink representation of the biolink object
    :return: a neo4j relationship
    """
    properties = {key: value for key, value in biolink_association.__dict__.items() if key != 'id'}
    return Relationship(biolink_subject,
                        get_pythonic_name(biolink_association.__class__.__name__),
                        biolink_object,
                        **properties)
