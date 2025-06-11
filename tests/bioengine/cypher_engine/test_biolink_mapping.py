from unittest import TestCase

from biolinkmodel.datamodel import OrganismalEntity, Cell, CellularComponent, BiologicalProcess, MolecularActivity, \
    Disease, PhenotypicFeature, ChemicalSubstance, NamedThing, Gene
from mock import patch
from parameterized import parameterized, param

from knowledge_engine.biolink_mapping import is_gene_mention, get_mapping
from knowledge_engine.models import Class, GO, MONDO


class TestMapping(TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Sets up the test class
        """
        cls.mapping = {
            OrganismalEntity: ['FMA', 'ZEBRAFISH_ANATOMICAL_ONTOLOGY', 'UBERON'],
            Cell: ['CL'],
            CellularComponent: ['CellularComponent'],
            BiologicalProcess: ['BiologicalProcess'],
            MolecularActivity: ['MolecularFunction'],
            Disease: ['DOID'],
            PhenotypicFeature: ['HUMAN_PHENOTYPE', 'CMPO'],
            ChemicalSubstance: ['CHEBI']
        }

    def test_is_gene_mention_when_exist_in_db(self):
        with patch('Bio.Entrez.read', return_value={'Count': '10'}):
            mention = is_gene_mention('STAT5B')
            assert mention is True

    def test_is_gene_mention_when_not_exist_in_db(self):
        with patch('Bio.Entrez.read', return_value={'Count': '0'}):
            mention = is_gene_mention('STAT5B')
            assert mention is False

    @parameterized.expand([
        param(ontology_prefix='FMA', expected=OrganismalEntity),
        param(ontology_prefix='DOID', expected=Disease),
        param(ontology_prefix='BOQ', expected=NamedThing),
    ])
    def test_mapping_with_namespace(self, ontology_prefix, expected):
        term = Class()
        term.ontology_prefix = ontology_prefix
        mapping = get_mapping(term, self.mapping)
        assert expected in list(mapping.keys())

    @parameterized.expand([
        param(annotation_has_obo_namespace='CellularComponent', expected=CellularComponent),
        param(annotation_has_obo_namespace='BiologicalProcess', expected=BiologicalProcess),
        param(annotation_has_obo_namespace='MolecularFunction', expected=MolecularActivity),
    ])
    def test_mapping_with_obo_namespace(self, annotation_has_obo_namespace, expected):
        term = GO()
        term.annotation_has_obo_namespace = annotation_has_obo_namespace
        mapping = get_mapping(term, self.mapping)
        assert expected in list(mapping.keys())

    def test_mapping_with_gene_mention(self):
        term = MONDO()
        term.label = 'STAT5B'
        mapping = get_mapping(term, self.mapping)
        assert Gene in list(mapping.keys())

    def test_get_relationship_node(self):
        pass