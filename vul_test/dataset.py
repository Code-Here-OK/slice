import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Dataset as DataSet
import pickle
from torch_geometric.data import Data
from torch_geometric.data.data import BaseData
import pandas as pd
import os
from tqdm import tqdm

class GraphDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    # def __getitem__(self, idx):
    #     node_embed, edge_matrix, target = self.data[idx]
    #     return torch.tensor(node_embed, dtype=torch.float), torch.tensor(edge_matrix, dtype=torch.float), torch.tensor(target, dtype=torch.long)
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        node_embed = torch.FloatTensor(sample['node_embed'])
        edge_matrix = torch.LongTensor(sample['edge_matrix'])
        target = torch.FloatTensor([sample['target']])
        return torch.tensor(node_embed, dtype=torch.float32), torch.tensor(edge_matrix, dtype=torch.float32), torch.tensor(target, dtype=torch.long)
    
class MyGraph(DataSet):
    def __init__(self, file_path):
         with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def len(self) -> int:
        return super().len()
    
    def __getitem__(self, idx):
        # print('__get__')
        sample = self.data.iloc[idx]
        node_embed = torch.FloatTensor(sample['node_embed'])
        edge_matrix = torch.LongTensor(sample['edge_matrix'])
        target = torch.FloatTensor([sample['target']])
        data = Data(x=node_embed, edge_index=edge_matrix, y=target)
        return data
    def get(self, idx: int) -> BaseData:
        return super().get(idx)
class MyGraphforNode(DataSet):
    def __init__(self, file_path):
         with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def len(self) -> int:
        return super().len()
    
    def __getitem__(self, idx):
        # print('__get__')
        sample = self.data.iloc[idx]
        node_embed = torch.FloatTensor(sample['node_embed'])
        edge_matrix = torch.LongTensor(sample['edge_matrix'])
        targets = torch.FloatTensor(sample['nodes_targets'])
        # list1 = [num.item() for num in targets]
        # target = torch.FloatTensor(list1)
        data = Data(x=node_embed, edge_index=edge_matrix, y=targets)
        return data
    def get(self, idx: int) -> BaseData:
        return super().get(idx)
    
class MyDataset(DataSet):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None, pre_filter=None):
        self.test = test
        self.filename = filename
        super().__init__(root, transform, pre_transform, pre_filter)
        
    def raw_file_names(self):
        return self.filename
    
    def processed_file_names(self):
        if self.raw_paths[0].endswith('pkl'):
            self.data = pd.read_pickle(self.raw_paths[0]).reset_index()
        elif self.raw_paths[0].endswith('csv'):
            self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]
    
    def download(self):
        pass
    
    def process(self):
        if self.raw_paths[0].endswith('pkl'):
            self.data = pd.read_pickle(self.raw_paths[0]).reset_index()
        elif self.raw_paths[0].endswith('csv'):
            self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        for index, value in tqdm(self.data.iterrows()):
            node_embed = torch.FloatTensor(value['node_embed'])
            edge_matrix = torch.LongTensor(value['edge_matrix'])
            target = torch.FloatTensor([value['target']])
            data = Data(x=node_embed, edge_index=edge_matrix, y=target)
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_test_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))
            
        
            
    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))        
        return data
        
    

    