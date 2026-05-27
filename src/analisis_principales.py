import pandas as pd

# DATAFRAME - TEST:
df_test = pd.read_csv('../data/train.csv')
print(df_test.head())

# DATRAIN - TRAIN:
df_train = pd.read_csv('../data/train.csv')
print(df_train.head())

print('========= Variables - Test =========')
print(df_test.dtypes)
print('========= Variables - Train =========')
print(df_train.dtypes)

print('========= Estadisticas - Test =========')
print(df_test.describe())
print('========= Estadisticas - Train =========')
print(df_train.describe())