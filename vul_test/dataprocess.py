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

def encode_node_features(nodes):                    ##  图的节点嵌入
    # codes = [node['properties']['CODE'] for node in nodes]
    # node_features = [encode_code(code.split(' ')) for code in codes]
    # node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    # return node_features_tensor
    nodes = nodes[:200] if len(nodes) > 200 else nodes
    codes = [node['properties']['CODE'] for node in nodes]
    # node_features = [encode_code(code.split(' ')) for code in codes]
    node_features = [encode_code(tokenizer_subword(code)) for code in codes]
    if len(node_features) < 200:
        zero_vector = [0] * len(node_features[0]) if node_features else [0] * 200  
        node_features += [zero_vector] * (200 - len(node_features))
    node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    return node_features_tensor
def encode_code(code):
    d2vmodel_path = '/data/SARD/d2vmodel/d2vmodel.model'
    model = Doc2Vec.load(d2vmodel_path)
    inferred_vector = model.infer_vector(code)
    return inferred_vector
def get_node_code(line):
    code = ''
    
    return code
def get_code_by_line(code_file, line):
    with open(code_file, 'r') as f:
        for index, code in enumerate(f):
            if line == index + 1:
                return code

def loadEdge(edge_path):
    with open(edge_path, 'r') as f:
        edges = json.load(f)
    return edges

def edge_index(edges, node_map):
    # 创建边邻接矩阵

    edge_index = [ [node_map[edge['startId']], node_map[edge['endId']]] for edge in edges ]
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    if edge_index_tensor.size(1) > 100:
        edge_index_tensor = edge_index_tensor[:, :100]
    elif edge_index_tensor.size(1) < 100:
        cols_to_add = 100 - edge_index_tensor.size(1)
        padding = torch.zeros(edge_index_tensor.size(0), cols_to_add, dtype=torch.long)
        edge_index_tensor = torch.cat((edge_index_tensor, padding), dim=1)
    return edge_index_tensor
def edge_type(edges):        #生成边类型的独热编码
    etypes = ['CFG', 'CDG', 'REACHING_DEF', 'AST', 'CALL']
    df = pd.DataFrame(columns=['id', 'startid', 'endid', 'type'])
    for edge in edges:
        id = edge['id']
        startid = edge['startId']
        endid = edge['endId']
        etype = edge['type'] 
        df = df.append({'id': id, 'startid': startid, 'endid': endid, 'type': etype}, ignore_index=True)
    edge_types = df['type'].unique()
    one_hot = pd.get_dummies(df['type'])
    if one_hot.shape[1] > 50:
        one_hot = one_hot.iloc[:, :50]
    elif one_hot.shape[1] < 50:
        cols_to_add = 50 - one_hot.shape[1]
    for i in range(cols_to_add):
        one_hot[f'extra_{i}'] = 0
    one_hot_tensor = torch.tensor(one_hot.values, dtype=torch.float)
    return one_hot_tensor
def set_target_graph(graph, target_path, slice_path):
    startLine, code_name, whole_uri, ruleId = reader.get_vul_info(graph)
    target_lines = reader.get_lines_by_targets(target_path + code_name.split('.')[0] + '.targets.txt')
    slice_lines = reader.get_lines_from_slice_txt(slice_path, graph)
    lines = set(target_lines + slice_lines)
    if startLine in lines:
        return 1
    else:
        return 0
    
    

if __name__ == '__main__':

    df = pd.DataFrame(columns=['node_embed', 'edge_matrix', 'edge_one_hot', 'target'])

    bad_graph_path ='/data/graph/bad/'
    bad_target_path = '/exp_data/exp_data/target_for_slice/target_bad/'
    bad_slice_path = '/exp_data/exp_data/slice_dir_0313/bad/'
    count = 0
    for graph in tqdm(os.listdir(bad_graph_path)[14783:]):
        nodes = loadNode(bad_graph_path + graph + '/nodes.json')
        edges = loadEdge(bad_graph_path + graph + '/edges.json')
        if len(nodes) != 0 and len(edges) != 0:
            node_map = node_index_map(nodes)
            node_embed = encode_node_features(nodes)
            edge_matrix = edge_index(edges, node_map)
            edge_one_hot = edge_type(edges)
            target = set_target_graph(graph, bad_target_path, bad_slice_path)
            # startLine, code_name, whole_uri, ruleId = reader.get_vul_info(graph)
            content = [node_embed, edge_matrix, edge_one_hot, target]
            with open('/workspace/cjl/temp/data_1.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(content)

            if target == 1:
                count += 1
            # df = df.append({'node_embed':node_embed, 'edge_matrix':edge_matrix, 'edge_one_hot':edge_one_hot, 'target':target}, ignore_index=True)
    print(count)  
    
             
    good_target_path = '/exp_data/exp_data/target_for_slice/target_good/'
    good_slice_path = '/exp_data/exp_data/slice_dir_0313/good/'
    good_graph_path = '/data/graph/good/'
    for graph in tqdm(os.listdir(good_graph_path)):
        nodes = loadNode(good_graph_path + graph + '/nodes.json')
        edges = loadEdge(good_graph_path + graph + '/edges.json')
        if len(nodes) != 0 and len(edges) != 0:
            node_map = node_index_map(nodes)                 #有问题
            node_embed = encode_node_features(nodes)
            edge_matrix = edge_index(edges, node_map)
            edge_one_hot = edge_type(edges)
            target = 0
            content = [node_embed, edge_matrix, edge_one_hot, target]
            with open('/temp/data_1.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(content)
