import pandas as pd

# Read the Excel file
df = pd.read_excel('predicted_companies.xlsx')

# Remove duplicates based on 'Company ID'
df_unique = df.drop_duplicates(subset='Company ID')

# Export the DataFrame without duplicates to a new Excel file
df_unique.to_excel('predicted_companies_unique.xlsx', index=False)
