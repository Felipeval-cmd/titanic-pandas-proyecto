import pandas as pd

# DATAFRAME - TEST:
df_test = pd.read_csv('../data/train.csv')
df_test.head()

# DATRAIN - TRAIN:
df_train = pd.read_csv('../data/train.csv')
df_train.head()
df_test = pd.read_csv('../data/test.csv')
df_test.head()

# DATAFRAME - TRAIN:
df_train = pd.read_csv('../data/train.csv')
df_train.head()

# DESCRIPCIONES:
print('========= Variables - Test =========')
print(df_test.dtypes)

print('========= Variables - Train =========')
print(df_train.dtypes)
