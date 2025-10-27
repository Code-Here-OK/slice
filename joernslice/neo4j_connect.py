from py2neo import *
from typing import List
import json
import subprocess
import os

# graph = Graph('bolt://localhost:7687', auth=('neo4j', '123456789'))
# graph = Graph('http://localhost:7474', auth=('neo4j', '123456789'))
def connect():
    graph = Graph("http://localhost:7474", auth = ("neo4j", "12345678"), name="neo4j")
    return graph
def show_whole_graph():

    graph = connect()
    query = '''
    Match (n)
    return properties(n) as properties, id(n) as id, labels(n) as label
    '''
    result = graph.run(query)
    return result

def get_nodes_by_lines(lines, directory, type):
    graph = connect()
    line_filters = ','.join(str(line) for line in lines)
    query1 = f'''
    match (n)
    where n.LINE_NUMBER in [{line_filters}]
    match (n)
    return id(n) as id, labels(n) as label, properties(n) as properties
    '''
    result = graph.run(query1).data()
    if not os.path.exists('/data/graph/' + type + '/' + directory):
        os.mkdir('/data/graph/' + type + '/' + directory)
    with open('/data/graph/' + type + '/' + directory + '/nodes.json', 'w') as f:
        f.write(json.dumps(result, indent=4))
    ids = [node ['id'] for node in result]   ## 节点列表
    get_edges_by_nodes_id(ids, directory, type)

    # return json.dumps(result)
def get_edges_by_nodes_id(ids, directory, type):
    graph = connect()
    ids = ','.join(str(id) for id in ids)
    query2 = f"""
        match (n)-[r]->(m)
        where id(n) in [{ids}] and id(m) in [{ids}]
        return distinct id(r) as id, type(r) as type, id(n) as startId, id(m) as endId, properties(r) as properties
        """
    result = graph.run(query2).data()
    if not os.path.exists('/data/graph/' + type + '/' + directory):
        os.mkdir('/data/graph/' + type + '/' + directory)
    with open('/data/graph/' + type + '/' + directory + '/edges.json', 'w') as f:
        f.write(json.dumps(result, indent=4))


def clear_all_data():
    graph = connect()
    query = """
        match(n)
        detach delete n
    """
    graph.delete_all()
def get_all():
    graph = connect()
    query = """
        match(n)
        return n
    """
    result = graph.run(query)
    return result

# if __name__ == '__main__':
#     print(get_all())
#     clear_all_data()
#     print(get_all())



