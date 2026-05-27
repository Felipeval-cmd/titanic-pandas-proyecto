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


# EDAD PROMEDIO:
edad_prom = df_junto['Age'].mean()
print(f'Edad Promedio de todo el dataset: {int(edad_prom.round())} años')

# SUPERVIVENCIA:
#Solo usamos train porque test no tiene Survived
sobrevivientes = df_train[df_train['Survived'] == 1].shape[0]
muertos = df_train[df_train['Survived'] == 0].shape[0]

print(f"Sobrevivieron: {sobrevivientes} pasajeros")
print(f"Murieron: {muertos} pasajeros")

# TARIFA PROMEDIO:
primera_class = df_junto[df_junto['Pclass'] == 1]['Fare'].mean()
print(f'Tarifa promedio en primera clase: {primera_class.round(2)} dolares')


# VIAJE - FAMILIARES:
# Sin familiares:
sin_familiares = df_junto[df_junto['Familiares'] == 0]
print(f'Numero de pasajeros sin familiares: {len(sin_familiares)}')

# Un solo familiar:
familiar_unico = df_junto[df_junto['Familiares'] == 1]
print(f'Personas que viajaron solo con UN familiar: {len(familiar_unico)}')

# Más de un familiar:
familiar_varios = df_junto[df_junto['Familiares'] > 1]
print(f'Personas que viajaron con MÁS DE UN familiar: {len(familiar_varios)}')


# EDADES MÍNIMA Y MÁXIMA:
edad_min = df_junto['Age'].min()
edad_max = df_junto['Age'].max()

print(f"Pasajero más joven: {edad_min} años")
print(f"Pasajero más viejo: {edad_max} años")

# EMBARQUE:
print(df_junto['Embarked'].value_counts())
print("\n===== Porcentaje =====\n")
s = df_junto['Embarked'].value_counts()
total = len(df_junto)
print(f"Southampton (S): {round(s['S']/total*100, 1)}%")
print(f"Cherbourg   (C): {round(s['C']/total*100, 1)}%")
print(f"Queenstown  (Q): {round(s['Q']/total*100, 1)}%")