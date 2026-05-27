import pandas as pd

# DATAFRAME CON LOS DATOS DE TEST Y TRAIN
df_test = pd.read_csv('../data/test.csv')
df_train = pd.read_csv('../data/train.csv')

df_test['Familiares'] = df_test['SibSp'] + df_test['Parch']
df_train['Familiares'] = df_train['SibSp'] + df_train['Parch']

df_test['Source'] = 'Test'
df_train['Source'] = 'Train'

# Como df_test no tiene la columna 'Survived' la agregamos como None
df_test['Survived'] = None
df_junto = pd.concat([df_test, df_train], ignore_index=True)



# CONTEO POR SEXO:
print("=== Conteo ===")
print(df_train.groupby('Sex')['Survived'].sum())
# Proporcionalmente
print("\n=== Proporción ===")
print(df_train.groupby('Sex')['Survived'].mean().round(2))

# CONTEO DE SUPERVIVENCIA POR NIÑOS Y HOMBRES ADULTOS:
ninos = df_train[df_train['Age'] < 18]
hombres_adultos = df_train[(df_train['Age'] >= 18) & (df_train['Sex'] == 'male')]
print("=== Conteo ===")
print(f"Niños sobrevivieron: {ninos['Survived'].sum()}")
print(f"Hombres adultos sobrevivieron: {hombres_adultos['Survived'].sum()}")
print("\n=== Proporción ===")
print(f"Tasa niños: {round(ninos['Survived'].mean()*100, 1)}%")
print(f"Tasa hombres adultos: {round(hombres_adultos['Survived'].mean()*100, 1)}%")



# CONTEO DE SUPERVIVENCIA POR EDAD:
mayores_50 = df_train[df_train['Age'] > 50]
menores_10 = df_train[df_train['Age'] < 10]
otros = df_train[(df_train['Age'] >= 10) & (df_train['Age'] <= 50)]
# Conteo
print("=== Conteo ===")
print(f"Mayores de 50 sobrevivieron: {mayores_50['Survived'].sum()}")
print(f"Menores de 10 sobrevivieron: {menores_10['Survived'].sum()}")
print(f"Otros sobrevivieron: {otros['Survived'].sum()}")
# Proporcionalmente
print("\n=== Proporción ===")
print(f"Tasa mayores de 50: {round(mayores_50['Survived'].mean()*100, 1)}%")
print(f"Tasa menores de 10: {round(menores_10['Survived'].mean()*100, 1)}%")
print(f"Tasa otros: {round(otros['Survived'].mean()*100, 1)}%")

# CONTEO DE PASAJEROS POR EMBARQUE:
# Conteo
print("=== Conteo ===")
print(df_train.groupby('Embarked')['Survived'].sum())
# Proporcionalmente
print("\n=== Proporción ===")
print((df_train.groupby('Embarked')['Survived'].mean()*100).round(1))
# ¿Qué relación hay en el puerto C por alto índice de supervivencia?
tabla = pd.crosstab(df_train['Embarked'], df_train['Pclass'], normalize='index').mul(100).round(1)
print(tabla)

# CONTEO DE SUPERVIVENCIA POR CLASE:
# Conteo
print("=== Conteo ===")
print(df_train.groupby('Pclass')['Survived'].sum())
# Proporcionalmente
print("\n=== Proporción ===")
print((df_train.groupby('Pclass')['Survived'].mean()*100).round(1))