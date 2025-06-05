
from py2neo.ogm import Property, RelatedTo, RelatedFrom

from bioengine.knowledge_engine.models.ols_models.ols_graph_object import Class


class DOID(Class):
        annotation_comment = Property("annotation-comment")
        annotation_id = Property("annotation-id")
        synonym = Property("synonym")
        obo_synonym = Property("obo_synonym")
        annotation_has_related_synonym = Property("annotation-has_related_synonym")
        obo_xref = Property("obo_xref")
        annotation_database_cross_reference = Property("annotation-database_cross_reference")
        obo_definition_citation = Property("obo_definition_citation")
        in_subset = Property("in_subset")
        annotation_has_obo_namespace = Property("annotation-has_obo_namespace")
        annotation_has_alternative_id = Property("annotation-has_alternative_id")
 
        children = RelatedTo('DOID')
        related = RelatedFrom('DOID')