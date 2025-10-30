import torch
from model import GCN, SimpleGGNN, GGNNModel
from dataset import MyDataset, MyGraph, MyGraphforNode
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import torch.nn.functional as F
import torch_geometric
from torch_geometric.explain import Explainer, GNNExplainer

device = torch.device('cuda:0')
# device = torch.device('cpu')

def train_node(model, train_loader, optimizer, loss_func):
    model.train()
    total_loss = 0
    step = 0
    count = 0
    for data in train_loader:
  
        data.to(device)
        optimizer.zero_grad() 
        _, lineout = model(data.x, data.edge_index, data.batch)
        # onehot_target = pd.get_dummies(data.y)
        # print(lineout.shape)
        # print(data.y.shape)
        # print(data.y)
        
        onehot_target = F.one_hot(data.y.to(torch.int64), num_classes=2)
        # onehot_target = torch.eye(2)[data.y.long(), :]
        loss = loss_func(lineout.to(torch.float32), onehot_target.to(torch.float32))
  
        total_loss += loss.item()
        step += 1
        loss.backward()
        optimizer.step()  
    return total_loss / step

def test_node(model, data_loader):
    model.eval()
    predictions = []
    true = []
    with torch.no_grad():
        for data in data_loader:
            data.to(device)
            _, lineout = model(data.x, data.edge_index, data.batch)
            pred = F.softmax(lineout, dim=1)
            pred_classes = pred.argmax(dim=1)

            # pred = F.sigmoid(out)
            # pred = (pred < 0.5).float()
            predictions.extend(pred_classes.tolist())
            true.extend(data.y.tolist())
        tn, fp, fn, tp = confusion_matrix(true, predictions).ravel()
        fnr = fn / (fn + tp)
        fpr = fp / (fp + tn)
        accuracy = accuracy_score(true, predictions)
        precision = precision_score(true, predictions)
        recall = recall_score(true, predictions)
        f1 = f1_score(true, predictions)
    return accuracy, precision, recall, f1, fnr, fpr
            
        
    

if __name__ == '__main__':
    
    print(torch.cuda.is_available())
    
    lr = 0.001
    epochs = 10
    batch_size = 4
    input_dim = 50
    hidden_dim = 64
    output_dim = 2
    num_layers = 5
    
    # model = GCN(hidden_dim=hidden_dim, input_dim=input_dim, output_dim=output_dim)
    # model = SimpleGGNN(hidden_dim=hidden_dim, input_dim=input_dim, output_dim=output_dim, num_layers=num_layers)
    model = GGNNModel(hidden_dim=hidden_dim, input_dim=input_dim, output_dim=output_dim, num_layers=num_layers, name='GGNNModel')

    model.to(device)
    # dataset = MyGraphforNode('/workspace/cjl/temp/new/data.pkl')
    # dataset = MyGraphforNode('/workspace/cjl/temp/new/nvd_over.pkl')  ###
    dataset = MyGraphforNode('/workspace/cjl/temp/new/nvd_new_over.pkl')



    train_set, val_test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=42)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=5e-4)
    # criterion = torch.nn.CrossEntropyLoss()
    pos_weight = torch.tensor([1]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
 
        
    ## 节点分类
    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = train_node(model, train_loader, optimizer, criterion)                                  ##节点分类

        print(f'epoch:{epoch} train_loss:{train_loss:.4f}')
        #验证
        accuracy, precision, recall, f1, fnr, fpr = test_node(model, val_loader)                                     ##节点分类
        print(f'Val - Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FNR: {fnr:.4f}, FPR: {fpr:.4f}')
        
        
    # model.eval()
    # explainer = Explainer(
    #     model=model,
    #     algorithm=GNNExplainer(epochs=200),
    #     explanation_type='model',
    #     node_mask_type='attributes',
    #     edge_mask_type='object',
    #     model_config=dict(
    #         mode='multiclass_classification',
    #         task_level='node',
    #         return_type='log_probs',
    #         ),
    # )
        
        #测试
    accuracy, precision, recall, f1, fnr, fpr = test_node(model, test_loader)                                                ##节点分类
    print(f'Test - Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FNR: {fnr:.4f}, FPR: {fpr:.4f}')

   
    