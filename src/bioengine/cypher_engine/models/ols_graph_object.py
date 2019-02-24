from py2neo.ogm import GraphObject, Property

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


class OlsClassGraphObject(GraphObject):
    id = Property()
    iri = Property()
    olsId = Property()
    short_form = Property()
    obo_id = Property()
    ontology_name = Property()
    has_children = Property()
    ontology_prefix = Property()
    description = Property()
    label = Property()
    is_defining_ontology = Property()
    is_root = Property()
    is_obsolete = Property()
    ontology_iri = Property()
    superClassDescription = Property()
    equivalentClassDescription = Property()


