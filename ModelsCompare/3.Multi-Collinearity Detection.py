import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# 读取数据
df = pd.read_excel('data\\enhanced_data.xlsx')

# 定义分类变量和连续变量
categorical_columns = ['Dual', 'Opinion', 'FinBack']
continuous_columns = ['GrossProfit', 'NetProfit', 'REC', 'Growth', 'NetProfitGrowth', 'FL', 'OL', 'CL', 'EPS', 'Top1', 'ROA1', 'Indep']

# 选择特征
X = df[categorical_columns + continuous_columns]

# 为VIF计算添加常数项
X_const = add_constant(X)

# 计算VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

# 计算总体平均VIF
average_vif = vif_data["VIF"][1:].mean()  # 排除常数项的VIF

# 将平均VIF添加到DataFrame
vif_data = vif_data._append({"Feature": "Average VIF", "VIF": average_vif}, ignore_index=True)

# 导出VIF数据到Excel
output_file = 'data\\vif_results.xlsx'
vif_data.to_excel(output_file, index=False)

print(f"VIF结果已经成功导出到 {output_file}")
