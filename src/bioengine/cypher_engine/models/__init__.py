from src.bioengine.cypher_engine.models.chebi_graph_object import CHEBI
from src.bioengine.cypher_engine.models.cmpo_graph_object import CMPO
from src.bioengine.cypher_engine.models.fma_graph_object import FMA
from src.bioengine.cypher_engine.models.mondo_graph_object import MONDO
from src.bioengine.cypher_engine.models.ro_graph_object import RO
from src.bioengine.cypher_engine.models.cl_graph_object import CL
from src.bioengine.cypher_engine.models.doid_graph_object import DOID
from src.bioengine.cypher_engine.models.go_graph_object import GO
from src.bioengine.cypher_engine.models.human_phenotype_graph_object import HUMAN_PHENOTYPE
from src.bioengine.cypher_engine.models.ols_graph_object import Class
from src.bioengine.cypher_engine.models.uberon_graph_object import UBERON
from src.bioengine.cypher_engine.models.zebrafish_anatomical_ontology_graph_object import ZEBRAFISH_ANATOMICAL_ONTOLOGY


class ModelFactory:

    @staticmethod
    def factory(namespace: str) -> Class:
        """
        A factory method for getting a model object with a specific namespace
        :param namespace: the ontology namespace
        :return: a model instance
        """
        model = None
        if namespace.lower() == 'doid':
            model = DOID()
        if namespace.lower() == 'go':
            model = GO()
        if namespace.lower() == 'cl':
            model = CL()
        if namespace.lower() == 'human_phenotype':
            model = HUMAN_PHENOTYPE()
        if namespace.lower() == 'uberon':
            model = UBERON()
        if namespace.lower() == 'zebrafish_anatomical_ontology':
            model = ZEBRAFISH_ANATOMICAL_ONTOLOGY()
        if namespace.lower() == 'mondo':
            model = MONDO()
        if namespace.lower() == 'chebi':
            model = CHEBI()
        if namespace.lower() == 'cmpo':
            model = CMPO()
        if namespace.lower() == 'fma':
            model = FMA()
        if namespace.lower() == 'ro':
            model = RO()
        return model
