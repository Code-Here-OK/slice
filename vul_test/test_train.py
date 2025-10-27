import torch
from dataset import MyGraph, MyDataset
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch import nn
from torch_geometric.nn import GatedGraphConv, global_mean_pool
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

class GGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim) -> None:
        super(GGNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        self.in_linear = nn.Linear(input_dim, hidden_dim)
        self.ggnn = GatedGraphConv(hidden_dim, num_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for _ in range(self.num_layers):
            x = self.ggnn(x.to(device), edge_index.to(device))
            x = self.relu(x)
        
        x = global_mean_pool(x, batch.to(device))
        x = self.linear(x)
        return torch.sigmoid(x).squeeze()



if __name__ == '__main__':
    
    print(torch.cuda.is_available())
    
    device = 'cuda'
    
    dataset = MyGraph('/workspace/cjl/temp/temp_1.pkl')
    # dataset = MyDataset(root='/workspace/cjl/temp', filename='/workspace/cjl/temp/temp_2.pkl')
    
    train_set, val_test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=42) 
    
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=True)

    model = GGNN(input_dim=50, hidden_dim=128, num_layers=3, output_dim=1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    num_epoth = 10
    
    for epoth in tqdm(range(1, num_epoth + 1)):
        train_loss = 0
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(data.y.to(device), out)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_loader)
        print(f'loss={train_loss:.4f}')
        
        model.eval()
        total_loss = 0
        predictions, true = [], []
        with torch.no_grad():
            for data in val_loader:
                out = model(data)
                loss = criterion(data.y.to(device), out)
                total_loss += loss.item()
                
                pred = (out > 0.5).float()  # Threshold the predictions
                predictions.extend(pred.tolist())
                true.extend(data.y.tolist())
        val_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(true, predictions)
        precision = precision_score(true, predictions)
        recall = recall_score(true, predictions)
        f1 = f1_score(true, predictions)
        print(f'Validation - Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    
    model.eval()
    total_loss = 0
    predictions, true = [], []
    with torch.no_grad():
        for data in test_loader:
            out = model(data)
            loss = criterion(data.y.to(device), out)
            
            total_loss += loss.item()
            pred = (out > 0.5).float()  # Threshold the predictions
            predictions.extend(pred.tolist())
            
            true.extend(data.y.tolist())
    print(predictions)
    print(true)
    test_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(true, predictions)
    precision = precision_score(true, predictions)
    recall = recall_score(true, predictions)
    f1 = f1_score(true, predictions)
    print(f'Test - Loss: {test_loss:.4f}, Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        
    
    
            
            
        
        
        
        
        
        
        
        

    
    
    
    
    
    