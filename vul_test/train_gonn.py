import torch
from model import GCN, SimpleGGNN, GGNNModel, GONN
from dataset import MyDataset, MyGraph, MyGraphforNode
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import torch.nn.functional as F
import torch_geometric
import numpy as np
import random

device = torch.device('cuda:0')
# device = torch.device('cpu')
nodeembed = []
nodetarget = []
def setup_seed(seed):
    # 设置随机种子，确保结果可复现
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(model, train_loader, optimizer, loss_func):
    model.train()
    total_loss = 0
    step = 0
    count = 0
    for data in train_loader:
  
        data.to(device)
        optimizer.zero_grad() 
        # print(data.x.shape)
        
        out, _ = model(data.x, data.edge_index, data.batch)

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
def train_node(model, train_loader, optimizer, loss_func):
    model.train()
    total_loss = 0
    step = 0
    count = 0
    for data in train_loader:
  
        data.to(device)
        optimizer.zero_grad() 
        print(data.x.shape)  # 注释掉
        _, lineout = model(data.x, data.edge_index, data.batch)
        # onehot_target = pd.get_dummies(data.y)
        print(lineout.shape)  # 注释掉
        # print(data.y.shape)
        # print(data.y)
        print(data.y.shape)  # 注释掉
        onehot_target = F.one_hot(data.y.to(torch.int64), num_classes=50)
        # onehot_target = torch.eye(2)[data.y.long(), :]
        # print(lineout.shape)
        # print(onehot_target.shape)
        loss = loss_func(lineout.to(torch.float32), onehot_target.to(torch.float32))
  
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
            out, _ = model(data.x, data.edge_index, data.batch)
            pred = F.softmax(out, dim=1)
            pred_classes = pred.argmax(dim=1)
            onehot_target = F.one_hot(data.y.to(torch.int64), num_classes=50)
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
def test_node(model, data_loader, loss_func):
    model.eval()
    predictions = []
    true = []
    total_loss = 0
    step = 0
    with torch.no_grad():
        for data in data_loader:
            data.to(device)
            _, lineout = model(data.x, data.edge_index, data.batch)
            pred = F.softmax(lineout, dim=1)
            pred_classes = pred.argmax(dim=1)

            # pred = F.sigmoid(lineout)
            # pred_classes = (pred > 0.5).to(torch.int)

            onehot_target = F.one_hot(data.y.to(torch.int64), num_classes=50)
            loss = loss_func(lineout.to(torch.float32), onehot_target.to(torch.float32))
            predictions.extend(pred_classes.tolist())
            true.extend(data.y.tolist())
            total_loss += loss.item()
            step += 1
        tn, fp, fn, tp = confusion_matrix(true, predictions).ravel()
        # print(f'tn={tn},fp={fp},fn={fn},tp={tp}')
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

    params = {'add_self_loops': False, 'agent_package':100, 'chunk_size':64, 'diff_or':True, 'dropout_rate':0.1, 'dropout_rate2':0.1, 'epochs':1, 'global_gating':True, 
            'hidden_channel':64, 'index_split': 0, 'learning_rate':0.001, 'log_freq':200, 'model':'ONGNN', 'num_layers':8, 'num_layers_input':1, 'patience':200, 'save_model':False, 'seed':42,
            'simple_gating':False, 'task':'nodesard', 'tm':True, 'weight_decay':5.0e-4, 'weight_decay2':None, 'out_channel': 50, 'in_channel': 50, 'batch_size':1, 'heads':4, 'gat':False }
    setup_seed(params['seed'])

    lr = params['learning_rate']
    epochs = params['epochs']
    batch_size = params['batch_size']
    # hidden_dim = 64
    # num_layers = 3
    ##早停参数
    best_val_loss = float('inf')
    patience = params['patience']
    improve_threshold = 0.995
    early_stopping_counter = 0
    best_model_state = None
    # model = GCN(hidden_dim=hidden_dim, input_dim=input_dim, output_dim=output_dim)
    # model = SimpleGGNN(hidden_dim=hidden_dim, input_dim=input_dim, output_dim=output_dim, num_layers=num_layers)
    # model = GGNNModel(hidden_dim=hidden_dim, input_dim=input_dim, output_dim=output_dim, num_layers=num_layers, name='GGNNModel')
    model = GONN(params=params, name='GONN')

    model.to(device)

    # dataset = MyGraph('/workspace/cjl/temp/new/data.pkl')      ##图分类数据集
    dataset = MyGraphforNode('/workspace/cjl/temp/new/data.pkl')  ##节点分类
    # dataset = MyGraph('/workspace/cjl/temp/new/nvd_new_over.pkl')


    train_set, val_test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=42)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    


    ##图分类
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=params['weight_decay'])
    # criterion = torch.nn.CrossEntropyLoss()
    
    # for epoch in tqdm(range(1, epochs + 1)):
    #     train_loss = train(model, train_loader, optimizer, criterion)                                     ##图分类

    #     print(f'epoch:{epoch} train_loss:{train_loss:.4f}')
    #     ##验证
    #     accuracy, precision, recall, f1, fnr, fpr, loss = test(model, val_loader, criterion)                                        ##图分类
    #     print(f'Val - Val_loss: {loss:.4f}, Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FNR: {fnr:.4f}, FPR: {fpr:.4f}')
    #     if loss < best_val_loss * improve_threshold:
    #         best_val_loss = loss
    #         early_stopping_counter = 0
    #         best_model_state = model.state_dict()
    #     else:
    #         early_stopping_counter += 1
    #         if early_stopping_counter >= patience:
    #             print(f'Early stopping at epoch {epoch}')
    #             break 
    # ##测试
    # accuracy, precision, recall, f1, fnr, fpr, _ = test(model, test_loader, criterion)                                                ##图分类
    # print(f'Test - Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FNR: {fnr:.4f}, FPR: {fpr:.4f}')

    
    


    print("开始训练")
    ##节点分类
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=params['weight_decay'])
    # criterion = torch.nn.CrossEntropyLoss()
    pos_weight = torch.tensor([15]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("进度中...")
    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = train_node(model, train_loader, optimizer, criterion)                                     ##节点分类

        print(f'epoch:{epoch} train_loss:{train_loss:.4f}')
        ##验证
        accuracy, precision, recall, f1, fnr, fpr, loss = test_node(model, val_loader, criterion)                                        ##节点分类
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
    accuracy, precision, recall, f1, fnr, fpr, _ = test_node(model, test_loader, criterion)                                                ##节点分类
    print(f'Test - Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FNR: {fnr:.4f}, FPR: {fpr:.4f}')

    
  

    


