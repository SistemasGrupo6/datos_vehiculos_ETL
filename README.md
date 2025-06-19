# Datos del SRI Vehículos del año 2024 ETL

Sistema completo de extracción, transformación y carga (ETL) para analizar los datos oficiales de vehículos nuevos registrados en Ecuador durante 2024, proporcionados por el Servicio de Rentas Internas (SRI). Incluye análisis exploratorio, visualizaciones interactivas y modelado predictivo básico.

## 🔧 Configuración del Entorno

Instalación automática de todas las librerías necesarias y configuración del entorno de trabajo en Google Colab.

```python
# Instalación de dependencias
!pip install pandas numpy matplotlib seaborn plotly scikit-learn gspread oauth2client openpyxl

# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

# Para conexión con Google Drive y Sheets
from google.colab import drive, files
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests
import io
import os
from datetime import datetime

print("✅ Librerías importadas correctamente")
print("📊 Configuración del entorno completada")
```

## 📁 Configuración de Google Drive

Monta Google Drive y crea el directorio de trabajo donde se almacenarán todos los archivos generados.

```python
# Montar Google Drive
drive.mount('/content/drive')
print("✅ Google Drive montado correctamente")

# Crear directorio de trabajo
work_dir = '/content/drive/MyDrive/SRI_Analisis_Vehiculos_2024'
os.makedirs(work_dir, exist_ok=True)
print(f"📁 Directorio de trabajo creado: {work_dir}")
```

## 📥 Descarga y Carga de Datos

Descarga automática de los datos del SRI desde la URL oficial con múltiples métodos de respaldo en caso de errores de conexión o encoding.

```python
def descargar_datos_sri():
    """
    Descarga los datos del SRI desde la URL oficial
    """
    url = "https://descargas.sri.gob.ec/download/datosAbiertos/SRI_Vehiculos_Nuevos_2024.csv"

    print("🔄 Descargando datos del SRI...")

    try:
        # Descargar el archivo
        response = requests.get(url)
        response.raise_for_status()

        # Guardar en Google Drive
        file_path = os.path.join(work_dir, 'SRI_Vehiculos_Nuevos_2024.csv')
        with open(file_path, 'wb') as f:
            f.write(response.content)

        print(f"✅ Datos descargados exitosamente: {file_path}")
        return file_path

    except Exception as e:
        print(f"❌ Error al descargar datos: {e}")
        return None

def cargar_datos_alternativo():
    """
    Función alternativa para cargar datos desde Google Drive
    """
    drive_file_id = "1sOxXIFxxDDFeWbib4UwdpSkm-kQ8eF-s"
    url = f"https://drive.google.com/uc?id={drive_file_id}"

    try:
        df = pd.read_csv(url)
        print("✅ Datos cargados desde Google Drive")
        return df
    except Exception as e:
        print(f"❌ Error al cargar desde Google Drive: {e}")
        return None

# Ejecutar descarga y carga
file_path = descargar_datos_sri()

if file_path and os.path.exists(file_path):
    try:
        df_vehiculos = pd.read_csv(file_path, encoding='utf-8')
        print("✅ Datos cargados desde descarga local con UTF-8")
    except UnicodeDecodeError:
        try:
            df_vehiculos = pd.read_csv(file_path, encoding='latin-1', engine='python', sep=';')
            print("✅ Datos cargados con Latin-1 y delimitador ';'")
        except Exception as e:
            print(f"❌ Error al cargar: {e}")
            df_vehiculos = None
else:
    df_vehiculos = cargar_datos_alternativo()
```

## 🔍 Exploración Inicial de Datos

Análisis preliminar del dataset para entender su estructura, dimensiones y características básicas.

```python
def explorar_datos_inicial(df):
    """
    Exploración inicial del dataset
    """
    print("=" * 60)
    print("📊 EXPLORACIÓN INICIAL DE DATOS")
    print("=" * 60)

    print(f"📏 Dimensiones del dataset: {df.shape}")
    print(f"📋 Número de filas: {df.shape[0]:,}")
    print(f"📋 Número de columnas: {df.shape[1]}")

    print("\n🔍 Información general del DataFrame:")
    print(df.info())

    print("\n📊 Primeras 5 filas:")
    print(df.head())

    print("\n📊 Estadísticas descriptivas:")
    print(df.describe(include='all'))

    print("\n🔍 Nombres de columnas:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")

    return df

# Ejecutar exploración
if 'df_vehiculos' in locals():
    df_vehiculos = explorar_datos_inicial(df_vehiculos)
```

## 🧹 Análisis de Calidad de Datos

Identificación de valores nulos, duplicados y problemas de calidad con visualizaciones correspondientes.

```python
def analisis_calidad_datos(df):
    """
    Análisis de calidad de datos
    """
    print("=" * 60)
    print("🔍 ANÁLISIS DE CALIDAD DE DATOS")
    print("=" * 60)

    # Valores nulos
    print("📊 Valores nulos por columna:")
    nulos = df.isnull().sum()
    porcentaje_nulos = (nulos / len(df)) * 100

    calidad_df = pd.DataFrame({
        'Columna': df.columns,
        'Valores_Nulos': nulos.values,
        'Porcentaje_Nulos': porcentaje_nulos.values,
        'Tipo_Dato': df.dtypes.values
    })

    print(calidad_df.sort_values('Porcentaje_Nulos', ascending=False))

    # Visualización de valores nulos
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 2, 1)
    sns.heatmap(df.isnull(), cbar=True, cmap='viridis')
    plt.title('Mapa de Valores Nulos')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 2)
    nulos_por_columna = df.isnull().sum()
    nulos_por_columna[nulos_por_columna > 0].plot(kind='bar')
    plt.title('Valores Nulos por Columna')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # Duplicados
    duplicados = df.duplicated().sum()
    print(f"\n📊 Registros duplicados: {duplicados}")

    return calidad_df

# Ejecutar análisis de calidad
if 'df_vehiculos' in locals():
    calidad_datos = analisis_calidad_datos(df_vehiculos)
```

## 📊 Análisis Exploratorio Completo

Análisis detallado de todas las variables categóricas y numéricas del dataset.

```python
def analisis_exploratorio_completo(df):
    """
    Análisis exploratorio completo
    """
    print("=" * 60)
    print("📊 ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("=" * 60)

    # Análisis por columnas categóricas
    columnas_categoricas = df.select_dtypes(include=['object']).columns

    print(f"📊 Columnas categóricas encontradas: {len(columnas_categoricas)}")
    for col in columnas_categoricas:
        print(f"\n🔍 Análisis de '{col}':")
        valores_unicos = df[col].nunique()
        print(f"   - Valores únicos: {valores_unicos}")

        if valores_unicos <= 20:
            print(f"   - Distribución:")
            print(df[col].value_counts().head(10))
        else:
            print(f"   - Top 10 valores más frecuentes:")
            print(df[col].value_counts().head(10))

    # Análisis por columnas numéricas
    columnas_numericas = df.select_dtypes(include=['int64', 'float64']).columns

    if len(columnas_numericas) > 0:
        print(f"\n📊 Columnas numéricas encontradas: {len(columnas_numericas)}")
        for col in columnas_numericas:
            print(f"\n🔍 Análisis de '{col}':")
            print(f"   - Min: {df[col].min()}")
            print(f"   - Max: {df[col].max()}")
            print(f"   - Media: {df[col].mean():.2f}")
            print(f"   - Mediana: {df[col].median():.2f}")

# Ejecutar análisis exploratorio
if 'df_vehiculos' in locals():
    analisis_exploratorio_completo(df_vehiculos)
```

## 📈 Visualizaciones Principales

Generación de gráficos estáticos completos incluyendo distribuciones por marca, año, tipo, cantón y análisis correlacional.

```python
def crear_visualizaciones_principales(df):
    """
    Crear visualizaciones principales del dataset
    """
    print("=" * 60)
    print("📊 CREANDO VISUALIZACIONES PRINCIPALES")
    print("=" * 60)

    plt.style.use('default')
    fig = plt.figure(figsize=(20, 24))

    # 1. Distribución por Marca
    if 'marca' in df.columns or 'MARCA' in df.columns:
        marca_col = 'marca' if 'marca' in df.columns else 'MARCA'
        plt.subplot(4, 2, 1)
        top_marcas = df[marca_col].value_counts().head(10)
        sns.barplot(x=top_marcas.values, y=top_marcas.index, palette='viridis')
        plt.title('Top 10 Marcas de Vehículos', fontsize=14, fontweight='bold')
        plt.xlabel('Cantidad')

        for i, v in enumerate(top_marcas.values):
            plt.text(v + 0.1, i, str(v), va='center')

    # 2. Distribución por Año
    if 'año' in df.columns or 'AÑO' in df.columns or 'ano' in df.columns:
        año_col = next((col for col in ['año', 'AÑO', 'ano'] if col in df.columns), None)
        if año_col:
            plt.subplot(4, 2, 2)
            df[año_col].value_counts().sort_index().plot(kind='bar', color='skyblue')
            plt.title('Distribución por Año', fontsize=14, fontweight='bold')
            plt.xlabel('Año')
            plt.ylabel('Cantidad')
            plt.xticks(rotation=45)

    # 3. Distribución por Tipo
    if 'tipo' in df.columns or 'TIPO' in df.columns:
        tipo_col = 'tipo' if 'tipo' in df.columns else 'TIPO'
        plt.subplot(4, 2, 3)
        df[tipo_col].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title('Distribución por Tipo de Vehículo', fontsize=14, fontweight='bold')
        plt.ylabel('')

    # 4. Distribución por Cantón
    if 'canton' in df.columns or 'CANTON' in df.columns:
        canton_col = 'canton' if 'canton' in df.columns else 'CANTON'
        plt.subplot(4, 2, 5)
        top_cantones = df[canton_col].value_counts().head(15)
        sns.barplot(x=top_cantones.values, y=top_cantones.index, palette='Set2')
        plt.title('Top 15 Cantones', fontsize=14, fontweight='bold')
        plt.xlabel('Cantidad')

    # Matriz de correlación
    columnas_numericas = df.select_dtypes(include=['int64', 'float64']).columns
    if len(columnas_numericas) > 1:
        plt.subplot(4, 2, 7)
        correlation_matrix = df[columnas_numericas].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlación', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Guardar visualizaciones
    plt.savefig(os.path.join(work_dir, 'visualizaciones_principales.png'),
                dpi=300, bbox_inches='tight')
    print("💾 Visualizaciones guardadas en Google Drive")

# Ejecutar visualizaciones
if 'df_vehiculos' in locals():
    crear_visualizaciones_principales(df_vehiculos)
```

## 🎯 Dashboard Interactivo

Creación de dashboard interactivo con Plotly que se exporta como archivo HTML para visualización dinámica.

```python
def crear_dashboard_interactivo(df):
    """
    Crear dashboard interactivo con Plotly
    """
    print("🎯 Creando dashboard interactivo...")

    columnas_disponibles = df.columns.tolist()
    print(f"Columnas disponibles: {columnas_disponibles}")

    # Crear subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribución Principal', 'Análisis Temporal',
                       'Comparación Categórica', 'Resumen'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "pie"}, {"type": "table"}]]
    )

    # Gráfico 1: Distribución de la primera columna categórica
    primera_categorica = df.select_dtypes(include=['object']).columns[0] if len(df.select_dtypes(include=['object']).columns) > 0 else df.columns[0]
    top_valores = df[primera_categorica].value_counts().head(10)

    fig.add_trace(
        go.Bar(x=top_valores.index, y=top_valores.values, name="Distribución"),
        row=1, col=1
    )

    # Gráfico 2: Tendencia temporal
    fig.add_trace(
        go.Scatter(x=df.index, y=df.index, mode='lines', name="Tendencia"),
        row=1, col=2
    )

    # Gráfico 3: Pie chart
    if len(df.select_dtypes(include=['object']).columns) > 1:
        segunda_categorica = df.select_dtypes(include=['object']).columns[1]
        pie_data = df[segunda_categorica].value_counts().head(5)
        fig.add_trace(
            go.Pie(labels=pie_data.index, values=pie_data.values, name="Distribución"),
            row=2, col=1
        )

    fig.update_layout(height=800, showlegend=False, title_text="Dashboard Interactivo - Análisis SRI Vehículos 2024")
    fig.show()

    # Guardar dashboard
    fig.write_html(os.path.join(work_dir, 'dashboard_interactivo.html'))
    print("💾 Dashboard interactivo guardado como HTML")

# Ejecutar dashboard
if 'df_vehiculos' in locals():
    crear_dashboard_interactivo(df_vehiculos)
```

## 🧽 Limpieza de Datos

Proceso completo de limpieza que incluye eliminación de duplicados, tratamiento de valores nulos y normalización de texto.

```python
def limpiar_datos(df):
    """
    Proceso de limpieza de datos
    """
    print("=" * 60)
    print("🧹 PROCESO DE LIMPIEZA DE DATOS")
    print("=" * 60)

    df_limpio = df.copy()
    print(f"📊 Registros iniciales: {len(df_limpio):,}")

    # 1. Eliminar duplicados
    duplicados_antes = df_limpio.duplicated().sum()
    df_limpio = df_limpio.drop_duplicates()
    duplicados_eliminados = duplicados_antes - df_limpio.duplicated().sum()
    print(f"🔄 Duplicados eliminados: {duplicados_eliminados}")

    # 2. Tratamiento de valores nulos
    print("\n🔍 Tratamiento de valores nulos:")
    for col in df_limpio.columns:
        nulos_col = df_limpio[col].isnull().sum()
        porcentaje_nulos = (nulos_col / len(df_limpio)) * 100

        if porcentaje_nulos > 50:
            print(f"   ❌ Columna '{col}' eliminada ({porcentaje_nulos:.1f}% nulos)")
            df_limpio = df_limpio.drop(columns=[col])
        elif porcentaje_nulos > 0:
            if df_limpio[col].dtype == 'object':
                df_limpio[col] = df_limpio[col].fillna('NO_ESPECIFICADO')
                print(f"   ✅ Columna '{col}' - nulos reemplazados con 'NO_ESPECIFICADO'")
            else:
                df_limpio[col] = df_limpio[col].fillna(df_limpio[col].median())
                print(f"   ✅ Columna '{col}' - nulos reemplazados con mediana")

    # 3. Normalizar texto
    for col in df_limpio.select_dtypes(include=['object']).columns:
        df_limpio[col] = df_limpio[col].astype(str).str.upper().str.strip()

    print(f"\n📊 Registros finales: {len(df_limpio):,}")
    print(f"📊 Columnas finales: {len(df_limpio.columns)}")

    # Reporte de limpieza
    reporte_limpieza = {
        'registros_iniciales': len(df),
        'registros_finales': len(df_limpio),
        'columnas_iniciales': len(df.columns),
        'columnas_finales': len(df_limpio.columns),
        'duplicados_eliminados': duplicados_eliminados,
        'porcentaje_datos_mantenidos': (len(df_limpio) / len(df)) * 100
    }

    return df_limpio, reporte_limpieza

# Ejecutar limpieza
if 'df_vehiculos' in locals():
    df_vehiculos_limpio, reporte_limpieza = limpiar_datos(df_vehiculos)
```

## 📊 Análisis Estadístico Avanzado

Análisis profundo de frecuencias, patrones temporales y distribuciones con visualizaciones correspondientes.

```python
def analisis_estadistico_avanzado(df):
    """
    Análisis estadístico avanzado
    """
    print("=" * 60)
    print("📊 ANÁLISIS ESTADÍSTICO AVANZADO")
    print("=" * 60)

    print("🔍 ANÁLISIS DE FRECUENCIAS POR CATEGORÍA:")
    columnas_categoricas = df.select_dtypes(include=['object']).columns

    for col in columnas_categoricas[:5]:
        print(f"\n📊 {col}:")
        frecuencias = df[col].value_counts()
        print(f"   - Categorías únicas: {len(frecuencias)}")
        print(f"   - Más frecuente: {frecuencias.index[0]} ({frecuencias.iloc[0]} registros)")
        print(frecuencias.head(10))

        # Visualización
        plt.figure(figsize=(12, 6))
        if len(frecuencias) <= 20:
            frecuencias.plot(kind='bar')
            plt.title(f'Distribución de {col}')
        else:
            frecuencias.head(20).plot(kind='bar')
            plt.title(f'Top 20 - Distribución de {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Análisis temporal
    columnas_temporales = [col for col in df.columns if any(palabra in col.lower() for palabra in ['año', 'fecha', 'mes', 'ano'])]

    if columnas_temporales:
        print(f"\n📅 ANÁLISIS TEMPORAL:")
        for col in columnas_temporales:
            distribución_temporal = df[col].value_counts().sort_index()
            print(f"   - Rango: {distribución_temporal.index.min()} - {distribución_temporal.index.max()}")

            plt.figure(figsize=(12, 6))
            distribución_temporal.plot(kind='line', marker='o')
            plt.title(f'Tendencia Temporal - {col}')
            plt.grid(True, alpha=0.3)
            plt.show()

# Ejecutar análisis avanzado
if 'df_vehiculos_limpio' in locals():
    analisis_estadistico_avanzado(df_vehiculos_limpio)
```

## 🤖 Modelado Predictivo Básico

Implementación de modelo Random Forest con encoding de variables categóricas y análisis de importancia de características.

```python
def preparar_modelo_basico(df):
    """
    Preparación básica para modelado
    """
    print("=" * 60)
    print("🤖 PREPARACIÓN PARA MODELADO")
    print("=" * 60)

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    df_modelo = df.copy()

    # Encoding de variables categóricas
    label_encoders = {}
    for col in df_modelo.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_modelo[col] = le.fit_transform(df_modelo[col].astype(str))
        label_encoders[col] = le

    print(f"✅ Variables categóricas codificadas: {len(label_encoders)}")

    if len(df_modelo.columns) > 2:
        # Variable objetivo y features
        target_col = list(label_encoders.keys())[0] if label_encoders else df_modelo.columns[0]
        feature_cols = [col for col in df_modelo.columns if col != target_col]

        X = df_modelo[feature_cols]
        y = df_modelo[target_col]

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar modelo
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        print(f"\n🎯 RESULTADOS DEL MODELO:")
        print(f"Variable objetivo: {target_col}")
        print(f"Accuracy: {model.score(X_test, y_test):.3f}")

        # Importancia de características
        if hasattr(model, 'feature_importances_'):
            importancias = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            print(f"\n📊 IMPORTANCIA DE CARACTERÍSTICAS:")
            print(importancias.head(10))

            # Visualizar importancias
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importancias.head(10), x='Importance', y='Feature')
            plt.title('Importancia de Características')
            plt.show()

        return model, label_encoders

    return None, label_encoders

# Ejecutar modelado
if 'df_vehiculos_limpio' in locals():
    model, label_encoders = preparar_modelo_basico(df_vehiculos_limpio)
```

## 📊 Exportación a Google Sheets

Exportación de los datos procesados a Google Sheets y generación de archivos CSV para análisis posterior.

```python
def conectar_google_sheets(df, sheet_name="SRI_Vehiculos_2024_Analisis"):
    """
    Conectar y exportar datos a Google Sheets
    """
    print("=" * 60)
    print("📊 CONEXIÓN CON GOOGLE SHEETS")
    print("=" * 60)

    try:
        # Configuración para Google Sheets API
        print("📝 Para conectar con Google Sheets, necesitas:")
        print("   1. Crear un proyecto en Google Cloud Console")
        print("   2. Habilitar Google Sheets API")
        print("   3. Crear credenciales de servicio")
        print("   4. Compartir la hoja con el email del servicio")

        # Alternativa: Guardar como CSV
        csv_path = os.path.join(work_dir, f"{sheet_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"💾 Datos guardados como CSV: {csv_path}")
        print("📋 Puedes importar este archivo manualmente a Google Sheets")

        # Ejemplo de código para Google Sheets
        """
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
        client = gspread.authorize(creds)
        
        sheet = client.create(sheet_name)
        worksheet = sheet.get_worksheet(0)
        worksheet.update([df.columns.values.tolist()] + df.values.tolist())
        """

    except Exception as e:
        print(f"❌ Error en conexión con Google Sheets: {e}")

# Ejecutar exportación
if 'df_vehiculos_limpio' in locals():
    conectar_google_sheets(df_vehiculos_limpio)
```

## 🚀 Uso del Sistema

1. **Ejecutar en Google Colab**: Cargar el notebook y ejecutar las celdas secuencialmente
2. **Autorizar Google Drive**: Permitir acceso cuando se solicite
3. **Datos automáticos**: El sistema descarga los datos del SRI automáticamente
4. **Análisis completo**: Se ejecuta todo el pipeline ETL automáticamente
5. **Resultados**: Visualizaciones y reportes se guardan en Google Drive

## 📋 Archivos Generados

- `visualizaciones_principales.png` - Gráficos estáticos principales
- `dashboard_interactivo.html` - Dashboard interactivo con Plotly  
- `SRI_Vehiculos_2024_Analisis.csv` - Dataset procesado y limpio
- Reportes de calidad de datos y limpieza en consola

## 🎯 Resultados Típicos

- **Análisis de marcas**: Top 10 marcas más registradas
- **Distribución temporal**: Patrones de registro por año/mes
- **Análisis geográfico**: Cantones con mayor registro de vehículos
- **Modelado predictivo**: Características más importantes para clasificación
- **Dashboard interactivo**: Visualizaciones dinámicas para exploración
