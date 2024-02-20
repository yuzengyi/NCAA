import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

# 加载数据
df = pd.read_excel('data\\processed_data.xlsx')

# 定义0,1变量和连续变量
categorical_columns = ['Dual', 'Opinion', 'FinBack']
continuous_columns = ['GrossProfit', 'NetProfit', 'REC', 'Growth', 'NetProfitGrowth', 'FL', 'OL', 'CL', 'EPS', 'Top1', 'ROA1', 'Indep']

# 分割特征和目标变量
X = df[categorical_columns + continuous_columns]
y = df['y']

# 数据标准化（对连续变量）
scaler = StandardScaler()
X[continuous_columns] = scaler.fit_transform(X[continuous_columns])

# 使用SMOTE算法
smote = SMOTE(sampling_strategy=1/3)  # 少数类与多数类的目标比例为1:3
X_res, y_res = smote.fit_resample(X, y)

# 查看经过SMOTE处理后的类别分布
print(f'原始数据集类别分布： {Counter(y)}')
print(f'经过SMOTE处理后的类别分布： {Counter(y_res)}')

# 创建处理后的DataFrame
df_res = pd.DataFrame(X_res, columns=categorical_columns + continuous_columns)
df_res['y'] = y_res

# 导出到Excel文件
output_file = 'data\\enhanced_data.xlsx'
df_res.to_excel(output_file, index=False)

print(f"增强后的数据已经成功导出到 {output_file}")
