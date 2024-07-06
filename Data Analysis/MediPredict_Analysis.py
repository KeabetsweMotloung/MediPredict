import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("Insurance_csv/insurance.csv")
df.head()

# check for columns that are null
null_values=df.isnull().sum()
null_values_rows = df.isnull().any(axis=1).sum()
unique=df.stack().unique()
data=df.info()
print(data)