from py2neo import Graph, cypher_escape, Table, NodeMatcher, RelationshipMatch

from meta_classes import Singleton
from settings import Config


class Connection(metaclass=Singleton):
    """
    A class that abstracts neo4j operations
    """
    def __init__(self, config: dict = None):
        if config is None:
            config = Config().get_property('neo4j')
        self.driver = Graph(config['uri'], auth=(config['user'], config['password']))
        self.node_matcher = NodeMatcher(self.driver)
        self.relation_matcher = RelationshipMatch(self.driver)

    def execute_string_query(self, query, **kwargs) -> Table:
        """
        A method that accepts a parameterized string, cleans the parameters, injects the params dynamically injects
        them and executes the query.
        :param query: a string representing a query with injectable parameters
        :param kwargs: keyed injectable parameters
        :return: a table representation of the data
        """
        args = [(x, cypher_escape(y)) for x, y in kwargs.items() if type(y) is str]
        args += [(x, y) for x, y in kwargs.items() if type(y) is not str]
        query_string = query().format(**dict(args))
        return self.driver.run(cypher=query_string).to_table()
