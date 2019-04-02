
from py2neo.ogm import Property

from src.bioengine.cypher_engine.models.ols_graph_object import Class


class MONDO(Class):
        annotation_comment = Property("annotation-comment")
        annotation_close_match = Property("annotation-closeMatch")
        annotation_id = Property("annotation-id")
        annotation_exact_match = Property("annotation-exactMatch")
        synonym = Property("synonym")
        annotation_see_also = Property("annotation-seeAlso")
        obo_synonym = Property("obo_synonym")
        annotation_has_related_synonym = Property("annotation-has_related_synonym")
        obo_xref = Property("obo_xref")
        annotation_database_cross_reference = Property("annotation-database_cross_reference")
        obo_definition_citation = Property("obo_definition_citation")
        in_subset = Property("in_subset")
        annotation_has_narrow_synonym = Property("annotation-has_narrow_synonym")
        annotation_has_broad_synonym = Property("annotation-has_broad_synonym")
        annotation_excluded_sub_class_of = Property("annotation-excluded subClassOf")
