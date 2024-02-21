import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 读取数据
df = pd.read_excel('data\\enhanced_data.xlsx')

categorical_columns = ['Dual', 'Opinion', 'FinBack']
continuous_columns = ['GrossProfit', 'NetProfit', 'REC', 'Growth', 'NetProfitGrowth', 'FL', 'OL', 'CL', 'EPS', 'Top1', 'ROA1', 'Indep']

# 定义特征和目标变量
X = df[categorical_columns + continuous_columns]
y = df['y']

# 数据集划分
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=2/3, random_state=42)

# 数据标准化处理
scaler = StandardScaler()
X_train_val_scaled = scaler.fit_transform(X_train_val)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 设置LogisticRegression和GridSearchCV的参数
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # 正则化强度的逆
    'penalty': ['l2'],  # 使用的惩罚
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],  # 对于小数据集，'liblinear'是个好选择
    'class_weight':['balanced']
}

# 创建LogisticRegression实例
log_reg = LogisticRegression()

# 创建GridSearchCV实例
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')

# 在训练+验证集上进行拟合
grid_search.fit(X_train_val_scaled, y_train_val)

# 输出最佳参数
print("最佳参数：", grid_search.best_params_)
