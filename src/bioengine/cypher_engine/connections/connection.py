from abc import ABC, abstractmethod
from meta_classes import Singleton
from py2neo import Table

from cypher_engine.models import ModelFactory


class Connection(ABC):
    """
    An Abstract class, abstracting Neo4j connections
    """

    @abstractmethod
    def execute_string_query(self, query, **kwargs) -> Table:
        """
        An abstract method for executing a cypher query
        :param query: a string representing a parameterized query
        :param kwargs: a dictionary of parameters
        """
        pass

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
