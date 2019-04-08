from cypher_engine.match import map_relation_with_ontology_terms
from cypher_engine.models import NamedThing, MolecularEntity, OrganismalEntity, \
    DiseaseOrPhenotypicFeature, ChemicalSubstance, AnatomicalEntityToAnatomicalEntityAssociation, \
    ChemicalToDiseaseOrPhenotypicFeatureAssociation, ChemicalToGeneAssociation, ChemicalToThingAssociation, \
    GeneToGeneAssociation, GeneToThingAssociation, ThingToDiseaseOrPhenotypicFeatureAssociation, \
    DiseaseOrPhenotypicFeatureAssociationToThingAssociation, Class
from preprocessor.extensions.noun_verb_noun import Relation
from collections import OrderedDict

test = Relation('Central diabetes insipidus', 'is', ['rare disease'], False)

terms = map_relation_with_ontology_terms(test)
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

print(terms)


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


possible_associations = get_association(relation=test, terms=terms)

# sorted(possible_associations.items(), key=lambda x: len(x[1][0]))
subject_biolink = list(get_mapping(list(terms.items())[0][1][0]).items())[0][0]
subject_term = terms[test.effector][0]
subject = subject_biolink(id=subject_term.iri,
                          name=subject_term.label,
                          category=[term.ontology_iri for term in terms[test.effector]],
                          iri=subject_term.iri,
                          full_name=subject_term.label,
                          synonym=subject_term.synonym if hasattr(subject_term, 'synonym') else [],
                          description=subject_term.description)
object_biolink = list(get_mapping(list(terms.items())[1][1][0]).items())[0][0]
object_term = terms[test.effectee[0]][0]
object = subject_biolink(id=subject_term.iri,
                         name=subject_term.label,
                         category=[term.ontology_iri for term in terms[test.effector]],
                         iri=subject_term.iri,
                         full_name=subject_term.label,
                         synonym=subject_term.synonym if hasattr(subject_term, 'synonym') else [],
                         description=subject_term.description)
knowledge_graph = list(possible_associations.items())[0][0](1, subject, test.relation, object, negated=test.negation)
print(knowledge_graph)

#
# object = subject_biolink(id=term.id,
#                     name=term.label,
#                     category=[term.ontology_iri for term in terms[effector]],
#                     iri=term.iri,
#                     full_name=term.label,
#                     synonym=term.synonym if hasattr(term, 'synonym') else [],
#                     description=term.description)
