import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 读取数据
df = pd.read_excel('data\\enhanced_data.xlsx')
# 定义分类变量和连续变量
categorical_columns = ['Dual', 'Opinion', 'FinBack']
continuous_columns = ['GrossProfit', 'NetProfit', 'REC', 'Growth', 'NetProfitGrowth', 'FL', 'OL', 'CL', 'EPS', 'Top1', 'ROA1', 'Indep']

# 定义特征和目标变量
X = df[categorical_columns + continuous_columns]
y = df['y']

# 数据集划分
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y,random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=(2/3), stratify=y_temp,random_state=42)

# 参数网格
param_grid = {
    'n_estimators': [100, 200, 300],  # 树的数量
    'max_depth': [None, 10, 20, 30],  # 树的最大深度
    'min_samples_split': [2, 5, 10],  # 内部节点再划分所需最小样本数
    'min_samples_leaf': [1, 2, 4],  # 叶节点最少样本数
    'max_features': ['auto', 'sqrt']  # 寻找最佳分割时要考虑的特征数量
}

# 初始化随机森林分类器
rfc = RandomForestClassifier(random_state=42)

# 使用GridSearchCV进行超参数优化
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best Parameters:", grid_search.best_params_)

# 使用最佳参数重新训练随机森林
best_rfc = grid_search.best_estimator_

# 评估模型
y_train_pred = best_rfc.predict(X_train)
y_val_pred = best_rfc.predict(X_val)
y_test_pred = best_rfc.predict(X_test)

print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred)}")
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred)}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred)}")
