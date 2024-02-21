import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 读取数据
df = pd.read_excel('data\\enhanced_data.xlsx')

categorical_columns = ['Dual', 'Opinion', 'FinBack']
continuous_columns = ['GrossProfit', 'NetProfit', 'REC', 'Growth', 'NetProfitGrowth', 'FL', 'OL', 'CL', 'EPS', 'Top1', 'ROA1', 'Indep']

# 定义变量
X = df[categorical_columns + continuous_columns]  # 选择适合的特征进行示例
y = df['y']

# 数据集划分
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=(2/3), random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 设置参数网格
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
}

# 初始化SVC模型
svc = SVC(kernel='rbf', class_weight='balanced')

# 进行网格搜索
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# 准备绘图数据
C, gamma = np.meshgrid(param_grid['C'], param_grid['gamma'])
accuracy = np.array(grid_search.cv_results_['mean_test_score']).reshape(3, 3)

# 绘制三维曲面图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(C, gamma, accuracy, cmap='viridis')

ax.set_xlabel('C', fontsize=12, fontname='Palatino Linotype')
ax.set_ylabel('Gamma', fontsize=12, fontname='Palatino Linotype')
ax.set_zlabel('Accuracy', fontsize=12, fontname='Palatino Linotype')
ax.set_title('SVM Hyperparameters Optimization', fontsize=14, fontname='Palatino Linotype')

plt.show()

# 输出最佳参数
print("最佳参数：", grid_search.best_params_)
