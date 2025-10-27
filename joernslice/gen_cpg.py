import os
import subprocess
from tqdm import tqdm
import reader
import json


## 生成cpg.bin
def gen_cpg_bin(code_path):
    index_path = '/slice/data/index.json'
    with open(index_path, 'r') as f:
        data = json.load(f)
    for directory in tqdm(os.listdir(code_path), desc='Generating cpg'):
        # vul_line, filename, whole_uri, cwe_id = reader.get_vul_info(directory)
        whole_uri = data[directory]['whole_uri']
        source_code = os.path.join(code_path, whole_uri)

        joern_parse_path = f"/slice/joern-cli/joern-parse"
        cpg_output_path = f"/slice/data/cpg_bin_all_data/{directory}"
        if not os.path.exists(cpg_output_path):
            os.makedirs(cpg_output_path)
        joern_parse_inst = joern_parse_path + ' ' + source_code + ' -o ' + cpg_output_path + '/cpg.bin'
        # print(joern_parse_inst)
        subprocess.run(joern_parse_inst, shell=True, stdout=subprocess.PIPE)
def neo4jcsv_export(bin_path):

    for directory in tqdm(os.listdir(bin_path), desc='generating neo4jcsv'):
        cpg_path = f"{bin_path}/{directory}/cpg.bin"
        joern_export_path = f"/slice/joern-cli/joern-export --repr=all --format=neo4jcsv"
        csv_output_path = f"/slice/data/cpg_csv/{directory}/"
        joern_export_inst = joern_export_path + ' ' + cpg_path + ' --out ' + csv_output_path
        subprocess.run(joern_export_inst, shell=True, stdout=subprocess.PIPE)


if __name__ == '__main__':
    
    # gen_cpg_bin('/expdata/all_data')
    # gen_cpg_bin('/expdata/all_data_without_comment')

    # neo4jcsv_export('/data/cpg_bin')
    # list1 = os.listdir('/expdata/all_data')
    bin_path = '/slice/data/cpg_bin_all_data'
    directory = '62540-v1.0.0'
    cpg_path = f"{bin_path}/{directory}/cpg.bin"
    joern_export_path = f"/joern-cli/joern-export --repr=all --format=neo4jcsv"
    csv_output_path = f"/data/cpg_csv/{directory}/"
    joern_export_inst = joern_export_path + ' ' + cpg_path + ' --out ' + csv_output_path
    subprocess.run(joern_export_inst, shell=True, stdout=subprocess.PIPE)

    print('hello')