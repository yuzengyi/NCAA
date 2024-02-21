import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
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
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=(2/9), random_state=42)

# 定义参数网格
param_grid = {
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'learning_rate': [0.01, 0.1, 0.2]
}

# 初始化XGBoost分类器
xgb = XGBClassifier(n_estimators=100)

# 使用GridSearchCV进行参数调优
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 打印最佳参数
print("最佳参数：", grid_search.best_params_)

# 使用最佳参数训练模型
best_xgb = grid_search.best_estimator_

# 对测试集和验证集进行评估
y_test_pred = best_xgb.predict(X_test)
y_val_pred = best_xgb.predict(X_val)

# 计算准确率
test_accuracy = accuracy_score(y_test, y_test_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Test Accuracy: {test_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
