
from py2neo.ogm import Property, RelatedTo, RelatedFrom

from bioengine.knowledge_engine.models.ols_models.ols_graph_object import Class


class RO(Class):
        annotation_editor_note = Property("annotation-editor note")
        annotation_comment = Property("annotation-comment")
        annotation_curator_note = Property("annotation-curator note")
        annotation_has_curation_status = Property("annotation-has curation status")
        annotation_expand_expression_to = Property("annotation-expand expression to")
        annotation_alternative_term = Property("annotation-alternative term")
        annotation_cites_as_authority = Property("annotation-citesAsAuthority")
        annotation_see_also = Property("annotation-seeAlso")
        synonym = Property("synonym")
        annotation_is_asymmetric_relational_form_of_process_class = Property("annotation-is asymmetric relational form of process class")
        annotation_editor_preferred_term = Property("annotation-editor preferred term")
        annotation_term_editor = Property("annotation-term editor")
        annotation_definition_source = Property("annotation-definition source")
        annotation_temporal_interpretation = Property("annotation-temporal interpretation")
        in_subset = Property("in_subset")
        annotation_obo_foundry_unique_label = Property("annotation-OBO foundry unique label")
        annotation_creator = Property("annotation-creator")
        annotation_is_homeomorphic_for = Property("annotation-is homeomorphic for")
        annotation_example_of_usage = Property("annotation-example of usage")
        annotation_creation_date = Property("annotation-creation_date")
        annotation_imported_from = Property("annotation-imported from")
        annotation_shorthand = Property("annotation-shorthand")
        annotation_created_by = Property("annotation-created_by")
 
        children = RelatedTo('RO')
        related = RelatedFrom('RO')