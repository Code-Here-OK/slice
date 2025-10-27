import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 加载数据
with open('/temp/temp.pkl', 'rb') as f:
    data = pickle.load(f)

node_embed = data['node_embed']
edge_matrix = data['edge_matrix']
edge_one_hot = data['edge_one_hot']

target = data['target']

# for index in range(0, 1996):
#     fea = torch.cat((node_embed[index].values, edge_matrix[index].values), 0)


X = (node_embed, edge_matrix, edge_one_hot)
# X = torch.cat((node_embed.values, edge_matrix.values), 0)
X = node_embed
y = target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



class GGNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GGNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, node_embed, edge_matrix, edge_one_hot):
        x = torch.cat((node_embed, edge_matrix, edge_one_hot), dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

input_size = 150  
hidden_size = 64
output_size = 2  
model = GGNN(input_size, hidden_size, output_size)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, X_train, y_train, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(*X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()


def test_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(*X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
        precision = precision_score(y_test.numpy(), predicted.numpy())
        recall = recall_score(y_test.numpy(), predicted.numpy())
        f1 = f1_score(y_test.numpy(), predicted.numpy())
        cm = confusion_matrix(y_test.numpy(), predicted.numpy())
        print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')
        print(f'Confusion Matrix:\n{cm}')


X_train = tuple(torch.tensor(data, dtype=torch.float) for data in X_train)
X_test = tuple(torch.tensor(data, dtype=torch.float) for data in X_test)
print(type(y_train))
print(type(y_test))
print(y_train[0])
print(y_test[0])
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)


train_model(model, X_train, y_train, criterion, optimizer, epochs=10)


test_model(model, X_test, y_test)