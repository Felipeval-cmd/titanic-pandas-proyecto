import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#DATAFRAME TEST:
df_test = pd.read_csv('../data/train.csv')

#DATAFRAME TRAIN:
df_train = pd.read_csv('../data/train.csv')

# DESCRIPCIONES:
print('========= Variables - Test =========')
print(df_test.dtypes)
print('========= Variables - Train =========')
print(df_train.dtypes)
print('========= Estadisticas - Test =========')
print(df_test.describe())
print('========= Estadisticas - Train =========')
print(df_train.describe())

# VARIABLE FAMILIARES:
df_test['Familiares'] = df_test['SibSp'] + df_test['Parch']
df_train['Familiares'] = df_train['SibSp'] + df_train['Parch']

# VARIABLE SOURCE:
df_test['Source'] = 'Test'
df_train['Source'] = 'Train'

#VARIABLE SURVIVED:
# Como df_test no tiene la columna 'Survived' la agregamos como None
df_test['Survived'] = None

# DATAFRAME JUNTO:
df_junto = pd.concat([df_test, df_train], ignore_index=True)

# TAMAÑOS Y PROPORCIONES:
print(f'Filas de train: {len(df_train)}')
print(f'Filas de test: {len(df_test)}')
print(f'Total de filas (con el df_junto): {len(df_junto)}')
p_test = round(len(df_test)/len(df_junto)*100)
print(f'Proporción de datos de test: {p_test}%')
p_train = round(len(df_train)/len(df_junto)*100)
print(f'Proporción de datos train: {p_train}%')

# VARIABLES NULAS:
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

# CLASIFICACION FAMILIAS Y CABINA:
df_junto[df_junto['Pclass'] == 3]['Cabin'].value_counts(dropna=False)
def clasificar_familia(cantidad):
    if cantidad == 0:
        return 'Grupo 1: Sin familiares'
    elif 1 <= cantidad <= 3:
        return 'Grupo 2: Familia pequeña'
    else:
        return 'Grupo 3: Familia grande'
df_junto['Categoria_Familiar'] = df_junto['Familiares'].apply(clasificar_familia)
print("\n=== DISTRIBUCIÓN INCLUYENDO PASAJEROS SIN CABINA (NaN) ===")
distribucion_completa = df_junto.groupby('Categoria_Familiar')['Cabin'].value_counts(dropna=False)
print(distribucion_completa)
# Contar cuántas cabinas (que no sean nulas) hay en cada grupo
resultado_cabinas = df_junto.groupby('Categoria_Familiar')['Cabin'].count()
print("Distribución de Cabinas por Grupo Familiar:")
print(resultado_cabinas)

# PORCNTAJE DE SUPERVIVENCIA POR GRUPOS:
df_junto['grupo_etario'] = df_junto['Age'].apply(
    lambda x: 'No Registra' if pd.isna(x) 
    else ('Niños' if x < 10 
          else ('Adolescentes' if x < 18 
                else ('Adultos' if x < 50 else 'Mayores')))
)
print("\n=== DISTRIBUCIÓN DE GRUPOS ETARIOS ===")
distribucion_etaria = df_junto['grupo_etario'].value_counts(dropna=False)
print(distribucion_etaria)


# Si es nulo (NaN) significa que NO tenía cabina. Si tiene texto, SÍ tenía.
df_junto['Tiene_Cabina'] = df_junto['Cabin'].apply(lambda x: 'No' if pd.isna(x) else 'Sí')
# Agrupamos por las tres variables y calculamos la proporción de supervivencia
proporciones = df_junto.groupby(['Sex', 'grupo_etario', 'Tiene_Cabina'])['Survived'].value_counts(normalize=True)*100
print("\n=== PROPORCIÓN DE SUPERVIVENCIA POR SEXO, EDAD Y CABINA (%) ===")
print(proporciones.to_frame(name='Porcentaje (%)'))

# GRAFICOS:
sns.set_theme(style="darkgrid")

# Filtramos df_junto para quitar los valores None o nulos de Survived
df_validos = df_junto[df_junto['Survived'].notna()]
df_validos = df_validos.copy()
df_validos['Survived'] = df_validos['Survived'].astype(int).astype(str)
plt.figure(figsize=(8, 5))
sns.histplot(data=df_validos, x="Age", hue="Survived", multiple="stack", palette="Set1")
plt.title("Distribución de Edad según Supervivencia")
plt.xlabel("Edad")
plt.ylabel("Cantidad de Pasajeros")
plt.show()


# 1. Filtramos los datos válidos para que no falle por los None de Survived
df_validos = df_junto[df_junto['Survived'].notna()].copy()
df_validos['Survived'] = df_validos['Survived'].astype(int).map({0: 'Falleció', 1: 'Sobrevivió'})
df_validos['Pclass'] = df_validos['Pclass'].map({1: '1ra Clase', 2: '2da Clase', 3: '3ra Clase'})
# 2. Creamos el gráfico con catplot
g = sns.catplot(
    data=df_validos, 
    x="Pclass", 
    hue="Survived", 
    col="Sex", 
    kind="count", 
    palette="Set1",
    order=['1ra Clase', '2da Clase', '3ra Clase']
)
g.set_axis_labels("Clase del Pasajero", "Cantidad de Personas")
g.set_titles("Género: {col_name}")
g._legend.set_title("Estado")
plt.show()

# Filtramos los datos válidos para evitar el error de los None
df_validos = df_junto[df_junto['Survived'].notna()].copy()
df_validos['Survived'] = df_validos['Survived'].astype(int).map({0: 'Falleció', 1: 'Sobrevivió'})
# Creamos el gráfico de cajas separado por género
g = sns.catplot(
    data=df_validos, 
    x="Survived", 
    y="Age", 
    col="Sex", 
    kind="box", 
    palette="Set2"
)
g.set_axis_labels("Estado", "Edad de los Pasajeros")
g.set_titles("Género: {col_name}")
plt.show()

# Filtramos los datos válidos
df_validos = df_junto[df_junto['Survived'].notna()].copy()
df_validos['Survived'] = df_validos['Survived'].astype(int).map({0: 'Falleció', 1: 'Sobrevivió'})

# Creamos el gráfico de violín separado por género
g = sns.catplot(
    data=df_validos, 
    x="Survived", 
    y="Age", 
    col="Sex", 
    kind="violin", 
    split=True, 
    palette="Pastel1"
)
g.set_axis_labels("Estado", "Edad")
g.set_titles("Género: {col_name}")
plt.show()