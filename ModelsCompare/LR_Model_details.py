import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# 读取数据
data = pd.read_excel('data\\enhanced_data.xlsx')

# 定义特征和目标变量
categorical_columns = ['Dual', 'Opinion', 'FinBack']
continuous_columns = ['GrossProfit', 'NetProfit', 'REC', 'Growth', 'NetProfitGrowth', 'FL', 'OL', 'CL', 'EPS', 'Top1', 'ROA1', 'Indep']
features = categorical_columns + continuous_columns
X = data[features]
y = data['y']

# 标准化连续变量
scaler = StandardScaler()
X[continuous_columns] = scaler.fit_transform(X[continuous_columns])

# 添加常数项
X_sm = sm.add_constant(X)

# 使用statsmodels的Logit类进行逻辑回归
sm_model = sm.Logit(y, X_sm)
result = sm_model.fit()

# 提取回归系数和统计信息
params = result.params
std_err = result.bse
conf_int = result.conf_int()
conf_int['OR'] = params
conf_int.columns = ['2.5%', '97.5%', 'OR']
conf_int = np.exp(conf_int)  # 计算Odds Ratio

# 计算Wald统计量和P值
wald_stat = (params / std_err) ** 2
p_values = result.pvalues

# 创建结果DataFrame
results_df = pd.DataFrame({
    'Coefficient': params,
    'Std.Err': std_err,
    'Wald': wald_stat,
    'P': p_values,
    'OR': np.exp(params),
    '2.5%': np.exp(conf_int['2.5%']),
    '97.5%': np.exp(conf_int['97.5%'])
})

# 重新格式化P值，添加标识符
results_df['P'] = results_df['P'].apply(lambda x: f'{x:.3f}***' if x < 0.001 else f'{x:.3f}**' if x < 0.01 else f'{x:.3f}*' if x < 0.05 else f'{x:.3f}')

# 导出到Excel
results_df.to_excel('logistic_regression_results.xlsx')

print('Logistic Regression model trained and results exported to Excel with coefficients.')
