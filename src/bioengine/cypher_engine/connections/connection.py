from abc import ABC, abstractmethod
from py2neo import Cursor, cypher_escape

from cypher_engine.models import ModelFactory


class Connection(ABC):
    """
    An Abstract class, abstracting Neo4j connections
    """

    def execute_string_query(self, query, **kwargs) -> Cursor:
        """
        An abstract method for executing a cypher query
        :param query: a string representing a parameterized query
        :param kwargs: a dictionary of parameters
        """
        args = [(x, cypher_escape(y)) for x, y in kwargs.items() if type(y) is str]
        args += [(x, y) for x, y in kwargs.items() if type(y) is not str]
        query_string = query().format(**dict(args))
        return self.driver.run(cypher=query_string)

    @abstractmethod
    def get_nodes(self, concept: str, node_query=None, factory: ModelFactory = None) -> list:
        """
        A method that obtains nodes from a neo4j database
        :param concept: the concept to look for
        :param node_query: a function for getting ontology in the form concepts x(concept: str) -> list
        :param factory: a factory of neo4j models
        :return: a list of qualified nodes
        """
        pass
