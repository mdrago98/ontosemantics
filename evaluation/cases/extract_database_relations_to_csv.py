from cypher_engine.connections.knowledge_graph_connection import KnowledgeGraphConnection

connection = KnowledgeGraphConnection()
query = """
MATCH (subject)-[association]->(obj)-[r:located_in]->(doc)
WHERE association.relation <> 'null' and obj.name <> 'null' and subject.name <> 'null' and type(association) <> 'could_refer_to' and type(association) <> 'located_in'
RETURN DISTINCT subject.name as subject, association.relation as predicate, obj.name as obj, type(association) as rel_type, doc.pmid as pmid
"""
df = connection.execute_string_query(query).to_data_frame()
cols = df.columns.tolist()
cols = [cols[4], cols[2], cols[0], cols[3]]
df = df[cols]
df.to_csv(path_or_buf='test.csv')
