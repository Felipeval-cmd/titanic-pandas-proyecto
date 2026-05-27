import pandas as pd

# DATAFRAME - TEST:
df_test = pd.read_csv('../data/train.csv')
df_test.head()

# DATRAIN - TRAIN:
df_train = pd.read_csv('../data/train.csv')
df_train.head()


# CREAR VARIABLES NUEVAS
df_test['Familiares'] = df_test['SibSp'] + df_test['Parch']
df_train['Familiares'] = df_train['SibSp'] + df_train['Parch']