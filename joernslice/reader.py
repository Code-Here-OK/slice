import json
import os
from tqdm import tqdm

def get_lines_from_txt(root_path, directory, type):
    lines = set()
    ##切片行
    txt_path = os.path.join(root_path, directory, 'slice_dir/' + type + '/DEBUG_slice_forward_df.txt')
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line, content in enumerate(f):
                content = content.split(':', 1)[1].strip()
                try:
                    data = json.loads(content)
                    lines.add(data['ln'])
                except json.decoder.JSONDecodeError:
                    pass
    txt_path = os.path.join(root_path, directory, 'slice_dir/' + type + '/DEBUG_slice_backward_df.txt')
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
    targets_txt = os.path.join(root_path, directory, type + '_targets.txt')
    with open(targets_txt, 'r') as f:
        for index, line in enumerate(f):
            line = line.split(':')[-1]
            lines.add(int(line))
    return list(lines)


def get_vul_info(dirid):
    index_path = '/root/expdata/index.json'
    with open(index_path, 'r') as f:
        data = json.load(f)
    return data[dirid]['line'], data[dirid]['code_file'], data[dirid]['whole_uri'], data[dirid]['ruleId']    
        

# if __name__ == '__main__':
#     all_data = '/root/expdata/all_data'

#     csv_path = '/root/joernSlice/data/cpg_csv_all_data'
    
#     lines = get_lines_from_txt(all_data, '231457-v2.0.0', 'bad')
#     print(lines)
#     line = []
#     with open('/root/joernSlice/src/joern-parse/test.txt', 'r') as f:
#         for index, line in enumerate(f):
#             line = line.split(':')[-1]
#             lines.add(int(line))
#     print(line)


