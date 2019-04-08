from py2neo import Graph, cypher_escape, Table, NodeMatcher, RelationshipMatch

from cypher_engine.models import Class, ModelFactory
from src.bioengine.cypher_engine.connections.connection import Connection
from settings import Config


class OntologyStoreConnection(Connection):
    """
    A class that abstracts neo4j operations
    """
    def __init__(self, config: dict = None):
        if config is None:
            config = Config().get_property('neo4j')['ontology_store']
        self.driver = Graph(config['uri'], auth=(config['user'], config['password']))
        self.node_matcher = NodeMatcher(self.driver)
        self.relation_matcher = RelationshipMatch(self.driver)

    def execute_string_query(self, query, **kwargs) -> Table:
        """
        A method that accepts a parameterized string, cleans the parameters, injects the params dynamically
        and executes the query.
        :param query: a string representing a query with injectable parameters
        :param kwargs: keyed injectable parameters
        :return: a table representation of the data
        """
        args = [(x, cypher_escape(y)) for x, y in kwargs.items() if type(y) is str]
        args += [(x, y) for x, y in kwargs.items() if type(y) is not str]
        query_string = query().format(**dict(args))
        return self.driver.run(cypher=query_string).to_table()

    def get_nodes(self, concept: str, node_query=None, factory: ModelFactory = None) -> list:
        """
        A method that enriches the nodes into their proper objects
        :param node_query: a function for getting ontology in the form concepts x(concept: str) -> list
        :param factory: a factory of neo4j models
        :param concept: a string representing the label of an ontology concept
        :return: a list of qualified nodes
        """
        if node_query is None:
            node_query = self.get_generic_nodes
        if factory is None:
            factory = ModelFactory
        nodes = []
        for node in node_query(concept):
            graph_object = factory.factory(node.ontology_name)
            if graph_object is not None:
                nodes.append(list(graph_object.match(self.driver).where(f"_.iri = '{node.iri}'")))
            else:
                nodes.append(node)
        return sum(nodes, [])

    def get_generic_nodes(self, concept: str) -> list:
        """
        :param concept: a string representation of the concept to look up
        :return: a node from an ontology
        """
        ols_node = Class()
        return list(ols_node.match(self.driver).where(f"_.label = '{concept}'"))
