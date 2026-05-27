
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


nulos_test = df_test.isnull().sum()
print(f'===== Datos faltantes en Test ======\n')
print(nulos_test)
print(f'\n===== Total: {nulos_test.sum()} =====')

nulos_train = df_train.isnull().sum()
print(f'===== Datos faltantes en Train ======\n')
print(nulos_train)
print(f'\n===== Total: {nulos_train.sum()} =====')

nulos_total = df_junto.isnull().sum()
print(f'===== Datos faltantes en Total ======\n')
print(nulos_total)
print(f'\n===== Total: {nulos_total.sum()} =====')

p_nulos = round((df_junto.isnull().sum()/ len(df_junto)) *100)
total = round((nulos_total.sum()/len(df_junto)) *100)
print(f'===== Porcentaje de datos faltantes totales =====\n')
print(p_nulos)
print(f'\n ===== Porcentaje total: {total} =====')