from py2neo.ogm import GraphObject, Property, RelatedTo, RelatedFrom

"""
id - Uniqiue id for the node in Neo4j
iri - International Resource Identifiers (IRI) for the class e.g. http://purl.obolibrary.org/obo/GO_0044832
olsId - A unique id for this IRI in a particular ontology created from the ontology name and the term IRI e.g. go:http://purl.obolibrary.org/obo/GO_0044832
short_form - short id for the term, usually the URI fragment e.g. GO_0044832
obo_id - OBO style id based on the short id e.g. GO:0044832
ontology_name - short unique name for the ontology e.g. go
has_children - convenient boolean to indicate if the node has child nodes
ontology_prefix - term id prefix e.g. GO
description - free text description of the term
label - unique label for the term
is_defining_ontology - boolean to indicate if this terms "belongs" in this ontology. e.g. A GO term in GO would be true, a PATO term in GO would be false
is_root - convenient boolean to indicate if the term is a root concept in the class hierarchy
is_obsolete - convenient boolean to indicate is the term is obsolete. Note Obsolete terms are also labelled with "Obsolete" in the Neo4j index
ontology_iri - unique identifier for the ontology e.g. http://purl.obolibrary.org/obo/go.owl
superClassDescription - A manchester syntax rendering of the logical superclass description with HTML markup
equivalentClassDescription - - A manchester syntax rendering of the logical equivalent class description with HTML markup
annotation-* - All annotation properties on a class are indexed in a dynamic filed. E.g. if a class has a has_obo_namespace annotation, in Neo4j this would be a property called annotation-has_obo_namespace
"""


class Class(GraphObject):
    """
    A Graph object modelling a generic OLS node
    """
    id = Property('id')
    iri = Property('iri')
    ols_id = Property('olsId')
    short_form = Property('short_form')
    obo_id = Property('obo_id')
    ontology_name = Property('ontology_name')
    has_children = Property('has_children')
    ontology_prefix = Property('ontology_prefix')
    description = Property('description')
    label = Property('label')
    is_defining_ontology = Property('is_defining_ontology')
    is_root = Property('is_root')
    is_obsolete = Property('is_obsolete')
    ontology_iri = Property('ontology_iri')
    super_class_description = Property('superClassDescription')
    equivalent_class_description = Property('equivalentClassDescription')

