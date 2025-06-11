
from py2neo.ogm import Property, RelatedTo, RelatedFrom

from bioengine.knowledge_engine.models.ols_models.ols_graph_object import Class


class GO(Class):
        annotation_comment = Property("annotation-comment")
        annotation_iao_0000231 = Property("annotation-IAO_0000231")
        annotation_id = Property("annotation-id")
        synonym = Property("synonym")
        obo_synonym = Property("obo_synonym")
        annotation_has_related_synonym = Property("annotation-has_related_synonym")
        annotation_consider = Property("annotation-consider")
        obo_xref = Property("obo_xref")
        obo_definition_citation = Property("obo_definition_citation")
        annotation_database_cross_reference = Property("annotation-database_cross_reference")
        in_subset = Property("in_subset")
        annotation_has_narrow_synonym = Property("annotation-has_narrow_synonym")
        annotation_has_broad_synonym = Property("annotation-has_broad_synonym")
        annotation_has_obo_namespace = Property("annotation-has_obo_namespace")
        annotation_term_replaced_by = Property("annotation-term replaced by")
        annotation_has_alternative_id = Property("annotation-has_alternative_id")
        annotation_creation_date = Property("annotation-creation_date")
        term_replaced_by = Property("term_replaced_by")
        annotation_created_by = Property("annotation-created_by")
 
        children = RelatedTo('GO')
        related = RelatedFrom('GO')