import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 加载数据
data = pd.read_excel('LSTM.xlsx')

# 数据预处理
data.fillna(method='ffill', inplace=True)
features = ['ROA1', 'GrossProfit', 'NetProfit', 'REC', 'Growth', 'NetProfitGrowth', 'FL', 'OL', 'CL', 'EPS', 'Indep',
            'Dual', 'Top1', 'Opinion', 'FinBack']
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# 构建用于LSTM的时间序列数据
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data.iloc[i-n_steps:i][features].values)
        y.append(data.iloc[i]['y'])
    return np.array(X), np.array(y)

n_steps = 3
X, y = create_sequences(data, n_steps)

# 数据集划分
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42)  # 0.125 x 0.8 = 0.1

# 创建PyTorch的Dataset
class TimeseriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

train_dataset = TimeseriesDataset(X_train, y_train)
val_dataset = TimeseriesDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return torch.sigmoid(out)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

        # 验证集性能
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                predicted = (outputs.squeeze() > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
            print(f'Validation Accuracy: {correct / total}')

# 参数调优
for hidden_dim in [50, 100]:
    for lr in [0.001, 0.01]:
        model = LSTMModel(input_dim=len(features), hidden_dim=hidden_dim)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print(f'Training model with hidden_dim: {hidden_dim}, learning_rate: {lr}')
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
