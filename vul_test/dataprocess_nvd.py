import sys
sys.path.append('/vul_test/utils')
from codeTokenizer import tokenizer_subword

import json
import os
import torch
import torch.nn.functional as F
from gensim.models.doc2vec import Doc2Vec
import pandas as pd
from tqdm import tqdm
sys.path.append('/joernslice')
import reader
import csv


def node_index_map(nodes):
    dict_node = {}
    for index, node in enumerate(nodes):
        node_id = node['id']
        dict_node[node_id] = index
    return dict_node
def loadNode(node_path):
    with open(node_path, 'r') as f:
        nodes = json.load(f)
    return nodes
def encode_node_features(nodes):
    codes = [node['properties']['CODE'] for node in nodes]
    node_features = [encode_code(tokenizer_subword(code)) for code in codes]
    node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    return node_features_tensor
def encode_code(code):
    d2vmodel_path = '/data/nvd/d2vmodel/d2vmodel.model'
    # d2vmodel_path = ''
    model = Doc2Vec.load(d2vmodel_path)
    inferred_vector = model.infer_vector(code)
    return inferred_vector  
def loadEdge(edge_path):
    with open(edge_path, 'r') as f:
        edges = json.load(f)
    return edges
def edge_index(edges, node_map):
    edge_index = [ [node_map[edge['startId']], node_map[edge['endId']]] for edge in edges ]
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index_tensor
def read_startline(cve_id):
    root = '/workspace/real_all/diff/lines'
    startlines = set()
    txt_path = os.path.join(root, cve_id + '.txt')
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for index, line in enumerate(f):
                # print(line)
                startlines.add(int(line))
    # print(startlines)
    return list(startlines)
def set_target_graph(nodes, graph, root_path):
    # startLine, code_name, whole_uri, ruleId = reader.get_vul_info(graph)
    # target_lines = reader.get_lines_by_targets(target_path)
    # slice_lines = reader.get_lines_from_slice_txt(slice_path)
    # lines = set(target_lines + slice_lines)
    startlines = read_startline(graph)           ##漏洞行
    lines = get_lines_from_txt(root_path, graph)
    nodes_targets = [1 if node['properties']['LINE_NUMBER'] in startlines else 0 for node in nodes]
    
    if 1 in nodes_targets:
        return 1, nodes_targets
    else:
        return 0, nodes_targets
    
def set_target_nodes(nodes, graph, target_path, slice_path):
    startLine, code_name, whole_uri, ruleId = reader.get_vul_info(graph)
    target_lines = reader.get_lines_by_targets(target_path + code_name.split('.')[0] + '.targets.txt')
    slice_lines = reader.get_lines_from_slice_txt(slice_path, graph)
    lines = set(target_lines + slice_lines)
    nodes_targets = [1 if node['properties']['LINE_NUMBER'] == startLine else 0 for node in nodes]
def get_lines_from_txt(root_path, cve):
    
    lines = set()
    ##切片行
    txt_path = os.path.join(root_path, 'slice', cve + '_vul/DEBUG_slice_forward_df.txt')
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line, content in enumerate(f):
                content = content.split(':', 1)[1].strip()
                try:
                    data = json.loads(content)
                    lines.add(data['ln'])
                except json.decoder.JSONDecodeError:
                    pass
    txt_path = os.path.join(root_path, 'slice', cve + '_vul/DEBUG_slice_backward_df.txt')
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line, content in enumerate(f):
                content = content.split(':', 1)[1].strip()
                try:
                    data = json.loads(content)
                    lines.add(data['ln'])
                except json.decoder.JSONDecodeError:
                    pass
    ##漏洞敏感词行
    targets_txt = os.path.join(root_path, 'lines', cve + '_vul.txt')
    if os.path.exists(targets_txt):
        with open(targets_txt, 'r') as f:
            for index, line in enumerate(f):
                line = line.split(':')[-1]
                lines.add(int(line))
    return list(lines)   
    

if __name__ == '__main__':
    
    data_root_path = '/workspace/real_all'
    graph_path ='/workspace/nvd_graph/'
    df = pd.DataFrame(columns=['node_embed', 'edge_matrix', 'target', 'nodes_targets'])
    count = 0
    count1 = 0
    for graph in tqdm(os.listdir(graph_path)):                                       ## cve_id
        print('loading nodes and edges')
        nodes = loadNode(graph_path + graph + '/nodes.json')
        edges = loadEdge(graph_path + graph + '/edges.json')
        if len(nodes) != 0 and len(edges) != 0:
            print('node features')
            node_map = node_index_map(nodes)
            node_embed = encode_node_features(nodes)
            edge_matrix = edge_index(edges, node_map)
            print('set targets')
            graph_target, nodes_targets = set_target_graph(nodes, graph, data_root_path)
            # print(graph_target)
            # print(nodes_targets)
            if 1 in nodes_targets:
                count1 += 1
            if graph_target == 1:
                count += 1
            df = df.append({'node_embed':node_embed, 'edge_matrix':edge_matrix, 'target':graph_target, 'nodes_targets':nodes_targets}, ignore_index=True)
    print(count)  
    print(count1)

    df.to_pickle('/temp/new/nvd_new.pkl')
