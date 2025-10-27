import json
import os
import subprocess
from tqdm import tqdm
from py2neo import *
import neo4jhandler  
from neo4j_admin_generator import generator
from unidiff import PatchSet

def get_lines_from_txt(root_path, cve):
    
    # root_path = 'root_path'
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
    with open(targets_txt, 'r') as f:
        for index, line in enumerate(f):
            line = line.split(':')[-1]
            lines.add(int(line))
    return list(lines)

def gen_cpg_bin():
    code_path = '/root/real/code'
    for cve in tqdm(os.listdir(code_path)):
        print(cve)
        cve_id = cve.split('_')[0]
        for file in os.listdir(os.path.join(code_path, cve)):
            if file.endswith('.c') or file.endswith('.cpp'):
                source_code = os.path.join(code_path, cve, file)
                joern_parse_path = f"/root/joernSlice/joern-cli/joern-parse"
                cpg_output_path = f"/root/joernSlice/nvd/cpg_bin/{cve_id}"
                if not os.path.exists(cpg_output_path):
                    os.makedirs(cpg_output_path)
                joern_parse_inst = joern_parse_path + ' ' + source_code + ' -o ' + cpg_output_path + '/cpg.bin'
                # print(joern_parse_inst)
                subprocess.run(joern_parse_inst, shell=True, stdout=subprocess.PIPE)
        
def neo4jcsv_export():
    bin_path = '/root/joernSlice/nvd/cpg_bin'
    for directory in tqdm(os.listdir(bin_path), desc='generating neo4jcsv'):
        cpg_path = f"{bin_path}/{directory}/cpg.bin"
        joern_export_path = f"/root/joernSlice/joern-cli/joern-export --repr=all --format=neo4jcsv"
        csv_output_path = f"/root/joernSlice/nvd/cpg_csv/{directory}/"
        joern_export_inst = joern_export_path + ' ' + cpg_path + ' --out ' + csv_output_path
        subprocess.run(joern_export_inst, shell=True, stdout=subprocess.PIPE) 
def connect():
    graph = Graph("http://localhost:7474", auth = ("neo4j", "12345678"), name="neo4j")
    return graph
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
    if not os.path.exists('/root/joernSlice/nvd/graph/' + directory):
        os.mkdir('/root/joernSlice/nvd/graph/' + directory)
    with open('/root/joernSlice/nvd/graph/' + directory + '/nodes.json', 'w') as f:
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
    if not os.path.exists('/root/joernSlice/nvd/graph/' + directory):
        os.mkdir('/root/joernSlice/nvd/graph/' + directory)
    with open('/root/joernSlice/nvd/graph/' + directory + '/edges.json', 'w') as f:
        f.write(json.dumps(result, indent=4))      
def clear_all_data():
    graph = connect()
    query = """
        match(n)
        detach delete n
    """
    graph.delete_all()


def get_diff(diff_root, vul_path, patch_path):
    for cve in tqdm(os.listdir(vul_path)):
        cve_id = cve.split('_')[0]
        diff_path = os.path.join(diff_root, cve_id)
        if not os.path.exists(diff_path):
            os.mkdir(diff_path)
        for file in os.listdir(os.path.join(vul_path, cve)):
            if file.endswith('.c') or file.endswith('.cpp'):
                vul_file = os.path.join(vul_path, cve, file)
                patch_file = os.path.join(patch_path, cve_id + '_patch', file)
                if os.path.exists(patch_file):
                    diff_inst = 'diff -u ' + vul_file + ' ' + patch_file + ' ' + '> ' + diff_path + '/' + cve_id + '.diff'
                    result = subprocess.run(diff_inst, shell=True, cwd=diff_path, stderr=subprocess.PIPE)
    
    
    
def diff_parse(diff_path):
    for cve in tqdm(os.listdir(diff_path)):
        cve_id = cve.split('_')[0]
        diff_file = os.path.join(diff_path, cve_id, cve_id + '.diff')
        lines = set()
        if os.path.exists(diff_file):
            with open(diff_file, 'r') as diff_file:
                diff_content = diff_file.read()
            # 解析diff文件
            patch_set = PatchSet(diff_content)
            # 遍历所有的补丁
            with open('/root/real/diff/lines/' + cve_id + '.txt', 'a+') as f:
                for patched_file in patch_set:
                    # print(f"File: {patched_file.path}")
                    for hunk in patched_file:
                        for line in hunk:
                            if line.is_removed:
                                f.write(str(line.source_line_no) + '\n')
                                # lines.add(line.source_line_no)
                            

if __name__ == "__main__":
    
    # gen_cpg_bin()
    # neo4jcsv_export()
    root_path = '/root/real'

    # print(get_lines_from_txt(root_path, 'CVE-2013-0848'))
    
    
    csv_path = '/root/joernSlice/nvd/cpg_csv'
    # for directory in tqdm(os.listdir(csv_path), desc='slicing......'):
    
    #     lines = get_lines_from_txt(root_path, directory)
    #     if len(lines) != 0:
    #         whole_csv_path = os.path.join(csv_path, directory)
    #         import_inst = generator(whole_csv_path)
    #         subprocess.run(import_inst, shell=True)
    #         neo4jhandler.neo4j_start()                                             ## 导入之前数据库必须关闭
    #         get_nodes_by_lines(lines, directory, 'bad')
    #         clear_all_data()                                         ## 清除数据库
    #         neo4jhandler.neo4j_stop()
    #     else:
    #         with open('/root/joernSlice/nvd/error.txt', 'a+') as f:
    #             f.write(directory + '\n')

        
        # lines = get_lines_from_txt(root_path, directory, 'good')
        # if len(lines) != 0:
        #     whole_csv_path = os.path.join(csv_path, directory)
        #     import_inst = generator(whole_csv_path)
        #     subprocess.run(import_inst, shell=True)
        #     neo4jhandler.neo4j_start()                                             ## 导入之前数据库必须关闭
        #     get_nodes_by_lines(lines, directory, 'good')
        #     clear_all_data()                                         ## 清除数据库  
        #     neo4jhandler.neo4j_stop()
        # else:
        #     with open('/root/joernSlice/data/exception/slice_good_error.txt', 'a+') as f:
        #         f.write(directory + '\n')
    
    vul_path = '/root/real/code'
    patch_path = '/root/real/patch'
    # list1 = os.listdir(vul_path)
    # list2 = os.listdir(patch_path)
    diff_path = '/root/real/diff/diff'
    # get_diff(diff_path, vul_path, patch_path)
    diff_parse('/root/real/diff/diff')
    print()