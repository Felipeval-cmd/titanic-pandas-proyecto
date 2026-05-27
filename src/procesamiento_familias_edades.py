# proyecto titanic - procesamiento de variables de familias y grupos etarios
import pandas as pd

df_test = pd.read_csv('../data/train.csv')
df_train = pd.read_csv('../data/train.csv')
df_test['Source'] = 'Test'
df_train['Source'] = 'Train'
df_junto = pd.concat([df_test, df_train], ignore_index=True)

df_test['Familiares'] = df_test['SibSp'] + df_test['Parch']
df_train['Familiares'] = df_train['SibSp'] + df_train['Parch']
df_test['Survived'] = None

# calculamos la variable familiares para que no de error
df_junto['Familiares'] = df_junto['SibSp'] + df_junto['Parch']

def clasificar_familia(cantidad):
    if cantidad == 0:
        return 'Grupo 1: Sin familiares'
    elif 1 <= cantidad <= 3:
        return 'Grupo 2: Familia pequeña'
    else:
        return 'Grupo 3: Familia grande'

# Aplicamos la función a la nueva columna
df_junto['Categoria_Familiar'] = df_junto['Familiares'].apply(clasificar_familia)

# Distribución incluyendo a las personas que NO tenían cabina (NaN)
print("\n=== DISTRIBUCIÓN INCLUYENDO PASAJEROS SIN CABINA (NaN) ===")
distribucion_completa = df_junto.groupby('Categoria_Familiar')['Cabin'].value_counts(dropna=False)
print(distribucion_completa)

# Contar cuántas cabinas (que no sean nulas) hay en cada grupo
resultado_cabinas = df_junto.groupby('Categoria_Familiar')['Cabin'].count()

print("Distribución de Cabinas por Grupo Familiar:")
print(resultado_cabinas)

# --- PORCENTAJE SUPERVIVENCIA SEGÚN CLASIFICACIÓN ---

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

# Agrupamos por las tres variables predictoras y calculamos la proporción de supervivencia
proporciones = df_junto.groupby(['Sex', 'grupo_etario', 'Tiene_Cabina'])['Survived'].value_counts(normalize=True) * 100

print("\n=== PROPORCIÓN DE SUPERVIVENCIA POR SEXO, EDAD Y CABINA (%) ===")
print(proporciones.to_frame(name='Porcentaje (%)'))
# Si es nulo (NaN) significa que NO tenía cabina. Si tiene texto, SÍ tenía.
df_junto['Tiene_Cabina'] = df_junto['Cabin'].apply(lambda x: 'No' if pd.isna(x) else 'Sí')

# Agrupamos por las tres variables predictoras y calculamos la proporción de supervivencia
proporciones = df_junto.groupby(['Sex', 'grupo_etario', 'Tiene_Cabina'])['Survived'].value_counts(normalize=True) * 100

print("\n=== PROPORCIÓN DE SUPERVIVENCIA POR SEXO, EDAD Y CABINA (%) ===")
print(proporciones.to_frame(name='Porcentaje (%)'))