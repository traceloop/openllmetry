import pandas as pd

# Test the exact data you mentioned
data = {'in_stock': [True, True, False, True]}
df = pd.DataFrame(data)

print("Column: in_stock")
print(f"Values: {df['in_stock'].tolist()}")
print(f"dtype: {df['in_stock'].dtype}")
print(f"is_bool_dtype: {pd.api.types.is_bool_dtype(df['in_stock'].dtype)}")
print(f"is_numeric_dtype: {pd.api.types.is_numeric_dtype(df['in_stock'].dtype)}")

# Test the logic from the code
dtype = df['in_stock'].dtype

if pd.api.types.is_bool_dtype(dtype):
    col_type = "BOOLEAN"
elif pd.api.types.is_numeric_dtype(dtype):
    col_type = "NUMBER"
else:
    col_type = "STRING"

print(f"Detected column type: {col_type}")
