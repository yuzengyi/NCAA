import pandas as pd

# 加载数据
df = pd.read_excel('data\\data.xlsx')  # 在Windows系统中使用反斜杠

# 分离y=1和y=0的数据行
df_y1 = df[df['y'] == 1]
df_y0 = df[df['y'] == 0].dropna()

# 对y=1的行进行填补
# 众数填充分类变量
for column in ['Dual', 'Opinion', 'FinBack']:
    mode = df_y1[column].mode()[0]
    df_y1[column].fillna(mode, inplace=True)

# 连续变量使用均值填补
continuous_columns = ['GrossProfit', 'NetProfit', 'REC', 'Growth', 'NetProfitGrowth', 'FL', 'OL', 'CL', 'EPS', 'Top1', 'ROA1', 'Indep']
df_y1[continuous_columns] = df_y1[continuous_columns].fillna(df_y1[continuous_columns].mean())

# 合并处理后的数据
df_processed = pd.concat([df_y1, df_y0], ignore_index=True)

# 确保没有缺失值
assert not df_processed.isnull().any().any(), "处理后的数据中仍有缺失值存在"

# 导出到Excel文件
output_file = 'data\\processed_data.xlsx'  # 在Windows系统中使用反斜杠
df_processed.to_excel(output_file, index=False)

print(f"数据已经成功导出到 {output_file}")
