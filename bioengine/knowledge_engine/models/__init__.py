import logging
from typing_extensions import deprecated
logger = logging.getLogger(__name__)

try:
    from bioengine.knowledge_engine.models.ols_models.chebi_graph_object import CHEBI
    from bioengine.knowledge_engine.models.ols_models.cmpo_graph_object import CMPO
    from bioengine.knowledge_engine.models.ols_models.fma_graph_object import FMA
    from bioengine.knowledge_engine.models.ols_models.mondo_graph_object import MONDO
    from bioengine.knowledge_engine.models.ols_models.ro_graph_object import RO
    from bioengine.knowledge_engine.models.ols_models.cl_graph_object import CL
    from bioengine.knowledge_engine.models.ols_models.doid_graph_object import DOID
    from bioengine.knowledge_engine.models.ols_models.go_graph_object import GO
    from bioengine.knowledge_engine.models.ols_models.human_phenotype_graph_object import HUMAN_PHENOTYPE
    from bioengine.knowledge_engine.models.ols_models.ols_graph_object import Class
    from bioengine.knowledge_engine.models.ols_models.uberon_graph_object import UBERON
    from bioengine.knowledge_engine.models.ols_models.zebrafish_anatomical_ontology_graph_object import ZEBRAFISH_ANATOMICAL_ONTOLOGY
except ImportError as e:
    logger.warn('Could not load one or more modules: ', e)



@deprecated("Use the local ontology manager instead")
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
