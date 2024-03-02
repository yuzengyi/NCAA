import pandas as pd

# 加载Excel文件
file_path = 'dataNeedClean.xlsx'  # 用您的文件路径替换这里
data = pd.read_excel(file_path)

# 保留'资不抵债'为1的行，即使其他列有空值；对于'资不抵债'不为1的行，如果有空值则删除
cleaned_data = data[(data['y'] == 1) | (data.notna().all(axis=1))]

# 将清洗后的数据导出到新的Excel文件
output_file = 'cleaned_data.xlsx'  # 用您希望保存的路径替换这里
cleaned_data.to_excel(output_file, index=False)
