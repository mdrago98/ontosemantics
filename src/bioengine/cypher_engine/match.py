from py2neo import NodeMatcher

from src.bioengine.cypher_engine import Connection
from src.bioengine.cypher_engine.models.ols_graph_object import OlsClassGraphObject

driver = Connection().driver


def get_nodes(list_concepts: list):
    """

    :param list_concepts: a list of concepts to look up
    :return: a list of neo4j nodes
    """
    ols_node = OlsClassGraphObject()
    matcher = NodeMatcher(driver)
    return list(matcher.match().where("_.label = 'myocardial infarction'"))
    # return ols_node.match(graph._driver).where(f"_.label =~ '{list_concepts[0]}.*'").first()


print(get_nodes(['myocardial']))
