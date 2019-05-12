
from py2neo.ogm import Property, RelatedTo, RelatedFrom

from src.bioengine.cypher_engine.models.ols_models.ols_graph_object import Class


class CHEBI(Class):
        annotation_iao_0000231 = Property("annotation-IAO_0000231")
        annotation_id = Property("annotation-id")
        annotation_smiles = Property("annotation-smiles")
        synonym = Property("synonym")
        obo_synonym = Property("obo_synonym")
        annotation_mass = Property("annotation-mass")
        annotation_has_related_synonym = Property("annotation-has_related_synonym")
        obo_xref = Property("obo_xref")
        annotation_database_cross_reference = Property("annotation-database_cross_reference")
        in_subset = Property("in_subset")
        annotation_has_obo_namespace = Property("annotation-has_obo_namespace")
        annotation_has_alternative_id = Property("annotation-has_alternative_id")
        annotation_inchikey = Property("annotation-inchikey")
        annotation_formula = Property("annotation-formula")
        annotation_iao_0100001 = Property("annotation-IAO_0100001")
        annotation_charge = Property("annotation-charge")
        annotation_monoisotopicmass = Property("annotation-monoisotopicmass")
        annotation_inchi = Property("annotation-inchi")
        term_replaced_by = Property("term_replaced_by")
 
        children = RelatedTo('CHEBI')
        related = RelatedFrom('CHEBI')