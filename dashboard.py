import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os

# =============================================================================
# CONFIGURACIÓN DE STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Dashboard - Monitoreo de Fatiga",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema oscuro/claro
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def cargar_datos():
    """Carga datos de encuesta y térmicos desde archivos si existen.
    Busca varios nombres comunes y hace fallback a datos de ejemplo.
    """
    df_encuesta = None

    posibles_encuestas = [
        'Monitoreo_de_Fatiga.xlsx',
        'monitoreo_de_fatiga.xlsx',
        'cuestionarios.csv',
        'cuestionarios.xlsx'
    ]

    for f in posibles_encuestas:
        if os.path.exists(f):
            try:
                if f.lower().endswith('.csv'):
                    df_encuesta = pd.read_csv(f)
                else:
                    df_encuesta = pd.read_excel(f)
                st.info(f"Cargando datos de encuesta desde: {f}")
                break
            except Exception as e:
                st.warning(f"Error leyendo {f}: {e}")

    if df_encuesta is None:
        # Datos de ejemplo si no hay archivo
        data_dict = {
            'Folio': ['F-001', 'F-002', 'F-003', 'F-004', 'F-005', 'F-006'],
            'Edad': [22, 25, 19, 28, 21, 24],
            'Sexo': ['Masculino', 'Femenino', 'Masculino', 'Femenino', 'Femenino', 'Masculino'],
            'Altura (en cm)': [175, 162, 180, 158, 165, 172],
            'Peso (en kg)': [75, 58, 82, 55, 62, 80],
            '¿En los últimos 3 días has dormido al menos 7 horas de forma regular?': ['Si', 'No', 'Si', 'No', 'No', 'Si'],
            '¿Cómo te sientes después del período de recuperación?': [
                'Muy recuperado(a), con energía para entrenar.', 
                'Recuperado(a), pero aún con algo de fatiga.', 
                'Muy recuperado(a), con energía para entrenar.',
                'Muy recuperado(a), con energía para entrenar.',
                'Muy recuperado(a), con energía para entrenar.',
                'Algo fatigado(a), me falta recuperar un poco más.'
            ]
        }
        df_encuesta = pd.DataFrame(data_dict)

    # Cargar datos térmicos
    datos_termicos = None
    posibles_termicos = ['Datos_Termicos.xlsx', 'datos_termicos.xlsx', 'datos_termicos.csv']
    for f in posibles_termicos:
        if os.path.exists(f):
            try:
                if f.lower().endswith('.csv'):
                    datos_termicos = pd.read_csv(f)
                else:
                    datos_termicos = pd.read_excel(f)
                st.info(f"Cargando datos térmicos desde: {f}")
                break
            except Exception as e:
                st.warning(f"Error leyendo {f}: {e}")

    if datos_termicos is None:
        datos_termicos = pd.DataFrame({
            'Folio': ['F-001', 'F-002', 'F-003', 'F-004', 'F-005', 'F-006'],
            'T_Max_Basal': [34.4, 33.5, 34.0, 33.8, 34.2, 33.9],
            'T_Max_Post': [34.2, 35.5, 33.8, 35.0, 35.2, 36.1],
            'T_Max_Rec': [35.3, 34.8, 34.5, 34.5, 34.9, 35.8],
            'Temp_Ambiente': [11.5, 12.0, 11.8, 12.5, 11.2, 12.1],
            'Humedad': [14.5, 15.0, 16.2, 14.8, 15.5, 16.0]
        })

    # Normalizar nombres de columna
    if 'Folio' not in df_encuesta.columns and 'folio' in df_encuesta.columns:
        df_encuesta = df_encuesta.rename(columns={'folio': 'Folio'})

    if 'Folio' not in datos_termicos.columns and 'folio' in datos_termicos.columns:
        datos_termicos = datos_termicos.rename(columns={'folio': 'Folio'})

    df_final = pd.merge(df_encuesta, datos_termicos, on='Folio', how='left')

    # Feature Engineering
    if 'Peso (en kg)' in df_final.columns and 'Altura (en cm)' in df_final.columns:
        df_final['IMC'] = df_final['Peso (en kg)'] / ((df_final['Altura (en cm)']/100) ** 2)
    else:
        df_final['IMC'] = np.nan

    le_sexo = LabelEncoder()
    if 'Sexo' in df_final.columns:
        df_final['Sexo'] = df_final['Sexo'].fillna('Desconocido')
        df_final['Sexo_Num'] = le_sexo.fit_transform(df_final['Sexo'])
    else:
        df_final['Sexo'] = 'Desconocido'
        df_final['Sexo_Num'] = 0

    if '¿En los últimos 3 días has dormido al menos 7 horas de forma regular?' in df_final.columns:
        df_final['Sueño_Bien'] = df_final['¿En los últimos 3 días has dormido al menos 7 horas de forma regular?'].apply(lambda x: 1 if str(x).strip().lower() in ['si','sí','s'] else 0)
    else:
        df_final['Sueño_Bien'] = 0

    # Deltas térmicos (si existen columnas)
    if 'T_Max_Post' in df_final.columns and 'T_Max_Basal' in df_final.columns:
        df_final['Delta_Ejercicio'] = df_final['T_Max_Post'] - df_final['T_Max_Basal']
    else:
        df_final['Delta_Ejercicio'] = np.nan

    if 'T_Max_Rec' in df_final.columns and 'T_Max_Post' in df_final.columns:
        df_final['Delta_Recuperacion'] = df_final['T_Max_Rec'] - df_final['T_Max_Post']
    else:
        df_final['Delta_Recuperacion'] = np.nan

    # Mapeo de fatiga
    mapa_fatiga = {
        'Muy recuperado(a), con energía para entrenar.': 0,
        'Recuperado(a), pero aún con algo de fatiga.': 1,
        'Neutral, ni muy descansado(a) ni muy cansado(a).': 1,
        'Algo fatigado(a), me falta recuperar un poco más.': 2,
        'Me siento triste y sin energía.': 2
    }

    if '¿Cómo te sientes después del período de recuperación?' in df_final.columns:
        df_final['Nivel_Fatiga'] = df_final['¿Cómo te sientes después del período de recuperación?'].map(mapa_fatiga).fillna(1)
    else:
        df_final['Nivel_Fatiga'] = 1

    return df_final

df_final = cargar_datos()

# =============================================================================
# SIDEBAR - NAVEGACIÓN
# =============================================================================
st.sidebar.header("📋 Navegación")
pagina = st.sidebar.radio(
    "Selecciona una sección:",
    ["📊 Resumen General", "👥 Datos de Atletas", "🌡️ Análisis Térmico", "🤖 Predicción de Fatiga"]
)

# =============================================================================
# PÁGINA 1: RESUMEN GENERAL
# =============================================================================
if pagina == "📊 Resumen General":
    st.header("📊 Resumen General")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Atletas", len(df_final))
    with col2:
        st.metric("Edad Promedio", f"{df_final['Edad'].mean():.1f} años")
    with col3:
        st.metric("IMC Promedio", f"{df_final['IMC'].mean():.1f}")
    with col4:
        buenos_sueno = (df_final['Sueño_Bien'].sum())
        st.metric("Con buen sueño", f"{buenos_sueno}/{len(df_final)}")
    
    st.divider()
    
    # Distribución de fatiga
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de Nivel de Fatiga")
        fatiga_counts = df_final['Nivel_Fatiga'].value_counts().sort_index()
        fatiga_labels = {0: '🟢 Bien', 1: '🟡 Moderado', 2: '🔴 Fatigado'}
        fatiga_counts.index = fatiga_counts.index.map(fatiga_labels)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        fatiga_counts.plot(kind='bar', ax=ax, color=colors)
        ax.set_ylabel('Cantidad de Atletas')
        ax.set_xlabel('Estado de Fatiga')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Distribución por Sexo")
        sexo_counts = df_final['Sexo'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#3498db', '#e91e63']
        sexo_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_ylabel('')
        st.pyplot(fig)

# =============================================================================
# PÁGINA 2: DATOS DE ATLETAS
# =============================================================================
elif pagina == "👥 Datos de Atletas":
    st.header("👥 Datos de Atletas")
    
    # Tabla interactiva
    st.subheader("Base de Datos Completa")
    
    # Seleccionar columnas a mostrar
    columnas_mostrar = st.multiselect(
        "Selecciona las columnas a mostrar:",
        df_final.columns.tolist(),
        default=['Folio', 'Edad', 'Sexo', 'Altura (en cm)', 'Peso (en kg)', 'IMC', 'Sueño_Bien']
    )
    
    st.dataframe(df_final[columnas_mostrar], use_container_width=True)
    
    # Estadísticas por género
    st.divider()
    st.subheader("Estadísticas por Género")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Masculino**")
        df_m = df_final[df_final['Sexo'] == 'Masculino']
        st.write(f"- Cantidad: {len(df_m)}")
        st.write(f"- Edad promedio: {df_m['Edad'].mean():.1f} años")
        st.write(f"- IMC promedio: {df_m['IMC'].mean():.1f}")
        st.write(f"- Con buen sueño: {df_m['Sueño_Bien'].sum()}/{len(df_m)}")
    
    with col2:
        st.write("**Femenino**")
        df_f = df_final[df_final['Sexo'] == 'Femenino']
        st.write(f"- Cantidad: {len(df_f)}")
        st.write(f"- Edad promedio: {df_f['Edad'].mean():.1f} años")
        st.write(f"- IMC promedio: {df_f['IMC'].mean():.1f}")
        st.write(f"- Con buen sueño: {df_f['Sueño_Bien'].sum()}/{len(df_f)}")

# =============================================================================
# PÁGINA 3: ANÁLISIS TÉRMICO
# =============================================================================
elif pagina == "🌡️ Análisis Térmico":
    st.header("🌡️ Análisis Térmico")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temperaturas Máximas por Fase")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(df_final))
        width = 0.25
        
        ax.bar(x - width, df_final['T_Max_Basal'], width, label='Basal (T1)', color='#3498db')
        ax.bar(x, df_final['T_Max_Post'], width, label='Post-Ejercicio (TF/T3)', color='#e74c3c')
        ax.bar(x + width, df_final['T_Max_Rec'], width, label='Recuperación (T5)', color='#2ecc71')
        
        ax.set_xlabel('Folio')
        ax.set_ylabel('Temperatura (°C)')
        ax.set_title('Evolución Térmica por Atleta')
        ax.set_xticks(x)
        ax.set_xticklabels(df_final['Folio'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Delta Térmico (Cambios de Temperatura)")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(df_final))
        width = 0.35
        
        ax.bar(x - width/2, df_final['Delta_Ejercicio'], width, label='Delta Ejercicio', color='#f39c12')
        ax.bar(x + width/2, df_final['Delta_Recuperacion'], width, label='Delta Recuperación', color='#9b59b6')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Folio')
        ax.set_ylabel('Cambio de Temperatura (°C)')
        ax.set_title('Deltas Térmicos por Atleta')
        ax.set_xticks(x)
        ax.set_xticklabels(df_final['Folio'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    # Condiciones ambientales
    st.divider()
    st.subheader("Condiciones Ambientales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_final['Folio'], df_final['Temp_Ambiente'], marker='o', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_xlabel('Folio')
        ax.set_ylabel('Temperatura (°C)')
        ax.set_title('Temperatura Ambiente')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_final['Folio'], df_final['Humedad'], marker='s', linewidth=2, markersize=8, color='#3498db')
        ax.set_xlabel('Folio')
        ax.set_ylabel('Humedad (%)')
        ax.set_title('Humedad Relativa')
        ax.grid(alpha=0.3)
        st.pyplot(fig)

# =============================================================================
# PÁGINA 4: PREDICCIÓN DE FATIGA
# =============================================================================
elif pagina == "🤖 Predicción de Fatiga":
    st.header("🤖 Modelo de Predicción de Fatiga")
    
    # Entrenar modelo
    features = [
        'Edad', 'Sexo_Num', 'IMC', 'Sueño_Bien',
        'T_Max_Basal', 'Delta_Ejercicio', 'Delta_Recuperacion',
        'Temp_Ambiente', 'Humedad'
    ]
    
    X = df_final[features]
    y = df_final['Nivel_Fatiga']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Importancia de variables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Importancia de Factores")
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        importances.plot(kind='barh', ax=ax, color='#3498db')
        ax.set_xlabel('Importancia')
        ax.set_title('Factores que Predicen Fatiga')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')
        ax.set_title('Matriz de Confusión')
        st.pyplot(fig)
    
    # Predicciones
    st.divider()
    st.subheader("Predicciones del Modelo")
    
    predicciones_df = pd.DataFrame({
        'Folio': df_final.loc[X_test.index, 'Folio'].values,
        'Predicción': y_pred,
        'Real': y_test.values,
        'Correcto': y_pred == y_test.values
    })
    
    predicciones_df['Predicción'] = predicciones_df['Predicción'].map({0: '🟢 Bien', 1: '🟡 Moderado', 2: '🔴 Fatigado'})
    predicciones_df['Real'] = predicciones_df['Real'].map({0: '🟢 Bien', 1: '🟡 Moderado', 2: '🔴 Fatigado'})
    
    st.dataframe(predicciones_df, use_container_width=True)
    
    # Precisión del modelo
    precision = (y_pred == y_test.values).sum() / len(y_test)
    st.metric("Precisión del Modelo", f"{precision:.0%}")
    
    # Interpretación
    st.divider()
    st.subheader("📈 Interpretación")
    st.info("""
    **¿Qué predice el modelo?**
    
    El modelo Random Forest analiza múltiples factores para clasificar el nivel de fatiga:
    - **Verde (0)**: Muy recuperado con energía para entrenar
    - **Amarillo (1)**: Recuperado pero con algo de fatiga
    - **Rojo (2)**: Algo fatigado, falta recuperar más
    
    **Factores clave:**
    Los gráficos de importancia muestran qué variables tienen mayor impacto en las predicciones.
    """)

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    Dashboard de Monitoreo de Fatiga | Instituto Tecnológico de Celaya | MCI 1° Semestre
    </div>
    """, unsafe_allow_html=True)
