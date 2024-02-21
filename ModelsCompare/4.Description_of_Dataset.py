import pandas as pd

# 读取数据
df = pd.read_excel('data\\enhanced_data.xlsx')

# 定义分类变量和连续变量
categorical_columns = ['Dual', 'Opinion', 'FinBack']
continuous_columns = ['GrossProfit', 'NetProfit', 'REC', 'Growth', 'NetProfitGrowth', 'FL', 'OL', 'CL', 'EPS', 'Top1', 'ROA1', 'Indep']
target_column = ['y']

# 对指定变量进行描述性统计
descriptive_stats = df[categorical_columns + continuous_columns + target_column].describe()

# 导出描述性统计结果到Excel
output_file = 'data\\descriptive_stats.xlsx'
descriptive_stats.to_excel(output_file)

print(f"描述性统计结果已经成功导出到 {output_file}")
