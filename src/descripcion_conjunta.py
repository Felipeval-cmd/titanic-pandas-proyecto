import pandas as pd

# DATAFRAME - TEST:
df_test = pd.read_csv('../data/train.csv')

# DATRAIN - TRAIN:
df_train = pd.read_csv('../data/train.csv')



df_test['Source'] = 'Test'
df_train['Source'] = 'Train'
# Como df_test no tiene la columna 'Survived' la agregamos como None
df_test['Survived'] = None

# DATAFRAME - JUNTO:
df_junto = pd.concat([df_test, df_train], ignore_index=True)
print(df_junto.describe())

# TAMAÑOS Y PROPORCIONES:
print(f'Filas de train: {len(df_train)}')
print(f'Filas de test: {len(df_test)}')
print(f'Total de filas (con el df_junto): {len(df_junto)}')
p_test = round(len(df_test)/len(df_junto)*100)
print(f'Proporción de datos de test: {p_test}%')
p_train = round(len(df_train)/len(df_junto)*100)
print(f'Proporción de datos train: {p_train}%')