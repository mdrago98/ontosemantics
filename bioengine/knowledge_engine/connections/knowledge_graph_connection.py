from warnings import deprecated

from py2neo import Graph, NodeMatcher, RelationshipMatch

from bioengine.knowledge_engine.models import ModelFactory
from settings import Config
from bioengine.knowledge_engine.connections import Connection

@deprecated('Use neo4j_knowledge_graph_connection or the new factory')
class KnowledgeGraphConnection(Connection):
    """
    A class abstracting a knowledge graph instance hosted in a neo4j graph
    """

    # def execute_string_query(self, query, **kwargs) -> Cursor:
    #     pass

    def get_nodes(self, concept: str, node_query=None, factory: ModelFactory = None) -> list:
        return list(self.driver.match(self.driver).where(f"_.label = '{concept}'"))

    def __init__(self, config: dict = None):
        if config is None:
            config = Config().get_property('neo4j')['knowledge_graph']
        self.driver = Graph(config['uri'], auth=(config['user'].strip(), config['password'].strip()))
        self.node_matcher = NodeMatcher(self.driver)
        self.relation_matcher = RelationshipMatch(self.driver)
