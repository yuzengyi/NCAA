import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据
data = pd.read_excel('LSTM.xlsx')

# 数据预处理
data.fillna(method='ffill', inplace=True)  # 向前填充缺失值
features = ['ROA1', 'GrossProfit', 'NetProfit', 'REC', 'Growth', 'NetProfitGrowth', 'FL', 'OL', 'CL', 'EPS', 'Indep', 'Dual', 'Top1', 'Opinion', 'FinBack']
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# 构建用于LSTM的时间序列数据
def create_sequences(data, n_steps):
    X, y, companies = [], [], []
    for company_id, group in data.groupby('id'):
        if len(group) < n_steps:
            continue
        for i in range(len(group) - n_steps):
            X.append(group.iloc[i:(i + n_steps)][features].values)
            y.append(group.iloc[i + n_steps]['y'])
            companies.append(company_id)
    return np.array(X), np.array(y), np.array(companies)

n_steps = 3
X, y, companies = create_sequences(data, n_steps)

# 创建一个包含公司ID和对应时间序列的DataFrame
company_sequences = pd.DataFrame({
    'CompanyID': companies,
    'Sequence': [x.tolist() for x in X],
    'Target': y
})

# 拆分训练集和测试集
train_data, test_data = train_test_split(company_sequences, test_size=0.3, random_state=42)

# 创建PyTorch的Dataset
class CompanyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data.iloc[idx]['Sequence'], dtype=torch.float32)
        y = torch.tensor(self.data.iloc[idx]['Target'], dtype=torch.float32)
        return x, y

train_dataset = CompanyDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, output_dim=1, num_layers=2):  # 修改hidden_dim为100
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

model = LSTMModel(input_dim=len(features))  # hidden_dim默认为100
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # learning_rate保持不变

# 修改train_model函数以保存每个epoch的损失值
def train_model(model, train_loader, criterion, optimizer, num_epochs=50):
    loss_values = []  # 用于存储每个epoch的损失值
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
        epoch_loss = total_loss / len(train_loader)
        loss_values.append(epoch_loss)  # 保存每个epoch的平均损失
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}')

    return loss_values  # 返回损失值列表
# 使用train_model函数训练模型并获取损失值
loss_values = train_model(model, train_loader, criterion, optimizer)


# 预测部分
test_sequences = torch.tensor(np.array(test_data['Sequence'].tolist()), dtype=torch.float32)
predictions = model(test_sequences).detach().numpy().flatten()
predicted_companies = test_data['CompanyID'][predictions > 0.5]

# 打印预测结果
print("Companies predicted to change 'y' from 0 to 1 in the next 3 years:")
print(predicted_companies.tolist())

# 导出结果
predicted_df = pd.DataFrame({'Company ID': predicted_companies, 'Prediction': 1})
predicted_df.to_excel('predicted_companies.xlsx', index=False)
# 使用Matplotlib绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Training Loss', color='blue')
plt.xlabel('Epochs', fontname='Palatino Linotype', fontsize=12)
plt.ylabel('Loss', fontname='Palatino Linotype', fontsize=12)
plt.title('Training Loss Over Epochs', fontname='Palatino Linotype', fontsize=14)
plt.legend()
plt.xticks(fontname='Palatino Linotype')
plt.yticks(fontname='Palatino Linotype')
plt.grid(True)
plt.savefig('Loss function curve.png')
plt.show()