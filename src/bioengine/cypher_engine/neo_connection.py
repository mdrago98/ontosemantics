from neo4j import GraphDatabase

from meta_classes import Singleton
from settings import Config
from .queries import all_ontologies as queries


class Connection(metaclass=Singleton):

    def __init__(self, config: dict = None):
        if config is None:
            config = Config().get_config('neo4j')
        self._driver = GraphDatabase.driver(config['uri'], auth=(config['user'], config['password']))

    def close(self):
        self._driver.close()

    def execute_query(self, query):
        with self._driver.session() as session:
            result = session.run(query)
        return result
