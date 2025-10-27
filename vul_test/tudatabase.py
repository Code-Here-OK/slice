import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        print(x.shape)
        x = self.conv1(x, edge_index)
        print(x.shape)
        x = x.relu()
        print(x.shape)
        x = self.conv2(x, edge_index)
        print(x.shape)
        x = x.relu()
        print(x.shape)
        x = self.conv3(x, edge_index)
        print(x.shape)

        x = global_mean_pool(x, batch)  
        print(x.shape)

        x = F.dropout(x, p=0.5, training=self.training)
        print(x.shape)
        x = self.lin(x)
        print(x.shape)
        
        return x
def train(model, train_loader):
    model.train()
    total_loss = 0
    step = 0
    for data in train_loader:  # I
        print(data)
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))  # 
        print(out)
        print(data.y)
        loss = criterion(out, data.y.to(device))  
        # print(f'{loss:.4f}')
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad() 
        
def test(model, test_loader):
    model.eval()
    correct = 0
    for data in test_loader:  
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))  
        pred = out.argmax(dim=1)  
        correct += int((pred == data.y.to(device)).sum())  
    return correct / len(test_loader.dataset) 

if __name__ == '__main__':
    
    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    # for data in dataset:
    #     print(data)
    # print(len(dataset))
    
    
    
    
    
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    train_data = dataset[0:150]
    test_data = dataset[150:]
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    
    device = ('cuda')
    model = GCN(hidden_channels=64)
    print(model.parameters)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
   
    
    epoch = 30
    for epoch in range(1, epoch + 1):
        train(model, train_loader)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        print(f'epoch = {epoch}, train_acc = {train_acc:.4f}, test_acc = {test_acc:.4f}')
        
        

    
