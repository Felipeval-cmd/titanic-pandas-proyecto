# proyecto titanic - puntos 12 y 13 completos (todos los graficos y analisis)

import seaborn as sns
import matplotlib.pyplot as plt

def generar_reporte_graficos(df_junto):
    # === GRAFICO 1: HISTOGRAMA DE EDAD ===
    # Configuramos el estilo de los gráficos
    sns.set_theme(style="darkgrid")

    # Filtramos df_junto para quitar los valores None o nulos de Survived
    df_validos = df_junto[df_junto['Survived'].notna()]

    # Forzamos a que la columna sea leída como texto/categoría (para evitar problemas de formato)
    df_validos = df_validos.copy()
    df_validos['Survived'] = df_validos['Survived'].astype(int).astype(str)

    # Creamos el gráfico usando los datos válidos
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df_validos, x="Age", hue="Survived", multiple="stack", palette="Set1")

    # Ponemos los títulos claros
    plt.title("Distribución de Edad según Supervivencia")
    plt.xlabel("Edad")
    plt.ylabel("Cantidad de Pasajeros")
    
    # Guardamos la imagen y cerramos el lienzo
    plt.savefig("../outputs/graficos/histograma_edad_supervivencia.png", dpi=300, bbox_inches='tight')
    plt.close()


    # === GRAFICO 2: BARRAS AGRUPADAS POR CLASE, SEXO Y SUPERVIVENCIA ===
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

    # 3. Ajustamos los títulos para que se vea impecable
    g.set_axis_labels("Clase del Pasajero", "Cantidad de Personas")
    g.set_titles("Género: {col_name}")
    g._legend.set_title("Estado")
    
    # Guardamos el catplot
    g.savefig("../outputs/graficos/barras_clase_sexo_supervivencia.png", dpi=300, bbox_inches='tight')
    plt.close()


    # === GRAFICO 3: BOXPLOT DE EDAD Y GÉNERO ===
    # Filtramos los datos válidos para evitar el error de los None
    df_validos = df_junto[df_junto['Survived'].notna()].copy()
    df_validos['Survived'] = df_validos['Survived'].astype(int).map({0: 'Falleció', 1: 'Sobrevivió'})

    # Creamos el gráfico de cajas separado por género
    g_box = sns.catplot(
        data=df_validos, 
        x="Survived", 
        y="Age", 
        col="Sex", 
        kind="box", 
        palette="Set2"
    )

    # Títulos claros
    g_box.set_axis_labels("Estado", "Edad de los Pasajeros")
    g_box.set_titles("Género: {col_name}")
    
    # Guardamos el boxplot
    g_box.savefig("../outputs/graficos/boxplot_edad_sexo_supervivencia.png", dpi=300, bbox_inches='tight')
    plt.close()


    # === GRAFICO 4: DIAGRAMA DE VIOLÍN ===
    # Filtramos los datos válidos
    df_validos = df_junto[df_junto['Survived'].notna()].copy()
    df_validos['Survived'] = df_validos['Survived'].astype(int).map({0: 'Falleció', 1: 'Sobrevivió'})

    # Creamos el gráfico de violín separado por género
    g_violin = sns.catplot(
        data=df_validos, 
        x="Survived", 
        y="Age", 
        col="Sex", 
        kind="violin", 
        split=True, 
        palette="Pastel1"
    )

    g_violin.set_axis_labels("Estado", "Edad")
    g_violin.set_titles("Género: {col_name}")
    
    # Guardamos el gráfico de violín
    g_violin.savefig("../outputs/graficos/violin_edad_sexo_supervivencia.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Todos los gráficos se generaron y guardaron en outputs/graficos/")

    # --- punto 13: conclusion final ---
    print("\n--- conclusion del analisis ---")
    print("las personas que tenian mas probabilidades de sobrevivir eran claramente las mujeres")
    print("y los ninos pequenos, cumpliendo con el protocolo de evacuacion. ademas, el factor")
    print("socioeconomico de tener una cabina asignada aumentaba drasticamente las opciones de salvarse.")