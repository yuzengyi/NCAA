import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load the Excel file
data = pd.read_excel('data\\enhanced_data.xlsx')
# 定义分类变量和连续变量
categorical_columns = ['Dual', 'Opinion', 'FinBack']
continuous_columns = ['GrossProfit', 'NetProfit', 'REC', 'Growth', 'NetProfitGrowth', 'FL', 'OL', 'CL', 'EPS', 'Top1', 'ROA1', 'Indep']

# 定义特征和目标变量
X = data[categorical_columns + continuous_columns]
y = data['y'].astype(int)  # Ensuring the target variable is an integer

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型初始化
models = {
    'Lasso Regularization': Lasso(),
    "Logistic Regression": LogisticRegression(C=0.01, class_weight='balanced', penalty='l2', solver='lbfgs', multi_class='multinomial'),
    "Decision Tree": DecisionTreeClassifier(criterion='gini', max_depth=5, max_leaf_nodes=8, min_samples_leaf=1, min_samples_split=2, splitter='best'),
    "Random Forest": RandomForestClassifier(max_depth=20, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=100),
    "XGBoost": XGBClassifier(colsample_bytree=1.0, gamma=0.5, learning_rate=0.2, max_depth=5, min_child_weight=1, subsample=0.6),
    "SVM": SVC(C=10, gamma=1, kernel='rbf', class_weight='balanced')
}

# 训练模型并获取特征重要性
feature_importances = pd.DataFrame(index=X.columns)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    if hasattr(model, 'feature_importances_'):
        feature_importances[name] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importances[name] = model.coef_[0]

# 保存特征重要性到Excel
feature_importances.to_excel('feature_importances.xlsx')

# 绘制决策树路径图
dt_model = models['Decision Tree']
plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=['0', '1'], max_depth=3)
plt.savefig('decision_tree.png')
plt.show()
