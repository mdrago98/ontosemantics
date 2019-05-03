
from py2neo.ogm import Property, RelatedTo, RelatedFrom

from cypher_engine.models.ols_models.ols_graph_object import Class


class UBERON(Class):
        annotation_editor_note = Property("annotation-editor note")
        annotation_homology_notes = Property("annotation-homology_notes")
        annotation_comment = Property("annotation-comment")
        annotation_id = Property("annotation-id")
        annotation_development_notes = Property("annotation-development_notes")
        annotation_fma_set_term = Property("annotation-fma_set_term")
        annotation_location_notes = Property("annotation-location_notes")
        synonym = Property("synonym")
        obo_synonym = Property("obo_synonym")
        annotation_has_related_synonym = Property("annotation-has_related_synonym")
        annotation_terminology_notes = Property("annotation-terminology_notes")
        obo_xref = Property("obo_xref")
        annotation_function_notes = Property("annotation-function_notes")
        annotation_database_cross_reference = Property("annotation-database_cross_reference")
        obo_definition_citation = Property("obo_definition_citation")
        in_subset = Property("in_subset")
        annotation_contributor = Property("annotation-contributor")
        annotation_has_narrow_synonym = Property("annotation-has_narrow_synonym")
        annotation_external_ontology_notes = Property("annotation-external_ontology_notes")
        annotation_dubious_for_taxon = Property("annotation-dubious_for_taxon")
        annotation_has_broad_synonym = Property("annotation-has_broad_synonym")
        annotation_has_obo_namespace = Property("annotation-has_obo_namespace")
        annotation_taxon_notes = Property("annotation-taxon_notes")
        annotation_depicted_by = Property("annotation-depicted_by")
        annotation_axiom_lost_from_external_ontology = Property("annotation-axiom_lost_from_external_ontology")
        annotation_has_alternative_id = Property("annotation-has_alternative_id")
        annotation_curator_notes = Property("annotation-curator notes")
        annotation_has_relational_adjective = Property("annotation-has_relational_adjective")
        annotation_external_definition = Property("annotation-external_definition")
 
        children = RelatedTo('UBERON')
        related = RelatedFrom('UBERON')