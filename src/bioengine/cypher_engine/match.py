from py2neo import NodeMatcher

from src.bioengine.cypher_engine import Connection
from src.bioengine.cypher_engine.models.doid_graph_object import doidGraphObject
from src.bioengine.cypher_engine.models.ols_graph_object import OlsClassGraphObject

driver = Connection().driver


def get_nodes(concept: str):
    """

    :param concept: a string representation of the concept to look up
    :return: a list of neo4j nodes
    """
    ols_node = doidGraphObject()
    matcher = NodeMatcher(driver)
    return list(matcher.match().where(f"_.label = '{concept}'"))
    # return ols_node.match(graph._driver).where(f"_.label =~ '{list_concepts[0]}.*'").first()


print(get_nodes('myocardial infarction'))
