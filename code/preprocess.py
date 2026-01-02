import pandas as pd
import os

#print("Current working directory:", os.getcwd())

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# loads the dataset, changes balnks to NaN
df = pd.read_csv('../data/heart_disease.csv', na_values='?', names = columns, header=0)
df_clean = df.dropna()
print("\nShape after dropping missing rows:", df_clean.shape) #amount of rows left

# Discretizes age
df_clean.loc[:, 'age_cat'] = pd.cut(
    df_clean['age'],
    bins=[0, 40, 60, 120],
    labels=['young', 'middle', 'old']
)

# Discretizes chol
df_clean.loc[:, 'chol_cat'] = pd.cut(
    df_clean['chol'],
    bins=[0, 200, 239, 600],
    labels=['low', 'normal', 'high']
)

# Discretizes trestbps
df_clean.loc[:, 'bp_cat'] = pd.cut(
    df_clean['trestbps'],
    bins=[0, 120, 139, 200],
    labels=['low', 'normal', 'high']
)

# Discretizes thalach (max heart rate)
df_clean.loc[:, 'thalach_cat'] = pd.cut(
    df_clean['thalach'],
    bins=[0, 100, 140, 250],
    labels=['low', 'normal', 'high']
)

# Discretizes oldpeak (ST depression)
df_clean.loc[:, 'oldpeak_cat'] = pd.cut(
    df_clean['oldpeak'],
    bins=[-1, 0, 2, 10],
    labels=['none', 'moderate', 'high']
)

# new groups
print("\nPreview of discretized features:")
print(df_clean[['age', 'age_cat', 'chol', 'chol_cat', 'trestbps', 'bp_cat', 'thalach', 'thalach_cat', 'oldpeak', 'oldpeak_cat']].head())

# shows some rows
#print(df.head())

# checks the missing values
#print("\nMissing values per column:")
#print(df.isnull().sum())
#print(df_clean.isnull().sum())

# saves new dataset
df_clean.to_csv('../data/heart_disease_clean.csv', index=False)

print("\nDone")
