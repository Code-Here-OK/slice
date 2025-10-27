from gen_cpg import *
from neo4j_admin_generator import *
import neo4jhandler
import os
import subprocess
import neo4j_connect
from reader import *
import logging

if __name__ == '__main__':
    
    all_data = '/root/expdata/all_data_without_comment'
    


    csv_path = '/root/joernSlice/data/cpg_csv_lyb'
    for directory in tqdm(os.listdir(csv_path)[0:6000], desc='slicing......'):
    
        
        
        lines = get_lines_from_txt(all_data, directory, 'bad')
        if len(lines) != 0:
            whole_csv_path = os.path.join(csv_path, directory)
            import_inst = generator(whole_csv_path)
            subprocess.run(import_inst, shell=True)
            neo4jhandler.neo4j_start()                                             ## 导入之前数据库必须关闭
            neo4j_connect.get_nodes_by_lines(lines, directory, 'bad')
            neo4j_connect.clear_all_data()                                         ## 清除数据库
            neo4jhandler.neo4j_stop()
        else:
            with open('/root/joernSlice/data/exception/slice_bad_error.txt', 'a+') as f:
                f.write(directory + '\n')

        
        lines = get_lines_from_txt(all_data, directory, 'good')
        if len(lines) != 0:
            whole_csv_path = os.path.join(csv_path, directory)
            import_inst = generator(whole_csv_path)
            subprocess.run(import_inst, shell=True)
            neo4jhandler.neo4j_start()                                             ## 导入之前数据库必须关闭
            neo4j_connect.get_nodes_by_lines(lines, directory, 'good')
            neo4j_connect.clear_all_data()                                         ## 清除数据库  
            neo4jhandler.neo4j_stop()
        else:
            with open('/root/joernSlice/data/exception/slice_good_error.txt', 'a+') as f:
                f.write(directory + '\n')
