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

device = torch.device('cuda:0')
# device = torch.device('cpu')


def train(model, train_loader, optimizer, loss_func):
    model.train()
    total_loss = 0
    step = 0
    count = 0
    for data in train_loader:
  
        data.to(device)
        optimizer.zero_grad() 
        out, _= model(data.x, data.edge_index, data.batch)
        # onehot_target = pd.get_dummies(data.y)
        onehot_target = F.one_hot(data.y.to(torch.int64), num_classes=2)
        # onehot_target = torch.eye(2)[data.y.long(), :]
        loss = loss_func(out.to(torch.float32), onehot_target.to(torch.float32))
        # print(loss)
        
        # loss = loss_func(out.to(torch.float32), data.y.to(torch.long))
        
        total_loss += loss.item()
        step += 1
        loss.backward()
        optimizer.step()  
    return total_loss / step


def test(model, data_loader, loss_func):
    model.eval()
    predictions = []
    true = []
    total_loss = 0
    step = 0
    with torch.no_grad():
        for data in data_loader:
            data.to(device)
            out, _= model(data.x, data.edge_index, data.batch)
            pred = F.softmax(out, dim=1)
            pred_classes = pred.argmax(dim=1)
            onehot_target = F.one_hot(data.y.to(torch.int64), num_classes=2)
            loss = loss_func(out.to(torch.float32), onehot_target.to(torch.float32))
  
            predictions.extend(pred_classes.tolist())
            true.extend(data.y.tolist())
            total_loss += loss.item()
            step += 1
        tn, fp, fn, tp = confusion_matrix(true, predictions).ravel()
        fnr = fn / (fn + tp)
        fpr = fp / (fp + tn)
        accuracy = accuracy_score(true, predictions)
        precision = precision_score(true, predictions)
        recall = recall_score(true, predictions)
        f1 = f1_score(true, predictions)
    return accuracy, precision, recall, f1, fnr, fpr, total_loss / step
        
    

if __name__ == '__main__':
    
    print(torch.cuda.is_available())
    
    input_dim = 50
    output_dim = 2

    lr = 0.0001
    epochs = 1000
    batch_size = 4
    hidden_dim = 64
    num_layers = 3
    
    ##早停参数
    best_val_loss = float('inf')
    patience = 5
    improve_threshold = 0.995
    early_stopping_counter = 0
    best_model_state = None
    # model = GCN(hidden_dim=hidden_dim, input_dim=input_dim, output_dim=output_dim)
    # model = SimpleGGNN(hidden_dim=hidden_dim, input_dim=input_dim, output_dim=output_dim, num_layers=num_layers)
    model = GGNNModel(hidden_dim=hidden_dim, input_dim=input_dim, output_dim=output_dim, num_layers=num_layers, name='GGNNModel')

    model.to(device)
    # dataset = MyGraph('/workspace/cjl/temp/data_all_over.pkl')
    # dataset = MyGraphforNode('/workspace/cjl/temp/data_all_targets_over.pkl')
    # dataset = MyGraph('/workspace/cjl/temp/new/data.pkl')
    dataset = MyGraph('/workspace/cjl/temp/new/nvd_new_over.pkl')


    train_set, val_test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=42)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=5e-4)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    
    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = train(model, train_loader, optimizer, criterion)                                     ##图分类

        print(f'epoch:{epoch} train_loss:{train_loss:.4f}')
        ##验证
        accuracy, precision, recall, f1, fnr, fpr, loss = test(model, val_loader, criterion)                                        ##图分类
        print(f'Val - Val_loss: {loss:.4f}, Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FNR: {fnr:.4f}, FPR: {fpr:.4f}')
        if loss < best_val_loss * improve_threshold:
            best_val_loss = loss
            early_stopping_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
    ##测试
    accuracy, precision, recall, f1, fnr, fpr, _ = test(model, test_loader, criterion)                                                ##图分类
    print(f'Test - Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FNR: {fnr:.4f}, FPR: {fpr:.4f}')

  