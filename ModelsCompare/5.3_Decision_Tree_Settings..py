import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 读取数据
df = pd.read_excel('data\\enhanced_data.xlsx')

# 定义分类变量和连续变量
categorical_columns = ['Dual', 'Opinion', 'FinBack']
continuous_columns = ['GrossProfit', 'NetProfit', 'REC', 'Growth', 'NetProfitGrowth', 'FL', 'OL', 'CL', 'EPS', 'Top1', 'ROA1', 'Indep']

# 定义特征和目标变量
X = df[categorical_columns + continuous_columns]
y = df['y']

# 假设X和y已经被定义，并且y中包含的是0和1的标签，比例为3比1
# 数据集划分
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=2/3, stratify=y_test, random_state=42)

# 参数网格
params = {
    "criterion": ("gini", "entropy"),
    "splitter": ("best", "random"),
    "max_depth": list(range(1, 20)),
    "min_samples_split": [2, 3, 4],
    "min_samples_leaf": list(range(1, 20)),
    "max_leaf_nodes": [3, 5, 6, 7, 8],  # 添加max_leaf_nodes参数
}

# 初始化决策树分类器
tree_clf = DecisionTreeClassifier(random_state=42)

# 使用GridSearchCV进行超参数优化
tree_cv = GridSearchCV(tree_clf, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=5)
tree_cv.fit(X_train, y_train)

# 输出最佳参数
best_params = tree_cv.best_params_
print(f"Best parameters: {best_params}")

# 使用最佳参数重新训练决策树
tree_clf = DecisionTreeClassifier(**best_params)
tree_clf.fit(X_train, y_train)

# 打印分数
def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        print(f"Train Result:\nAccuracy Score: {accuracy_score(y_train, pred)}\n")
    else:
        pred = clf.predict(X_test)
        print(f"Test Result:\nAccuracy Score: {accuracy_score(y_test, pred)}\n")

print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)
