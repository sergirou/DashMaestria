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
    """Carga datos desde Datos_Termicos.xlsx con mapeo automático de columnas."""
    
    # Intentar cargar desde archivo existente
    df = None
    archivos_posibles = ['Datos_Termicos.xlsx', 'datos_termicos.xlsx', 'Monitoreo_de_Fatiga.xlsx']
    
    for archivo in archivos_posibles:
        if os.path.exists(archivo):
            try:
                df = pd.read_excel(archivo)
                st.info(f"Cargando datos desde: {archivo}")
                break
            except Exception as e:
                st.warning(f"Error leyendo {archivo}: {e}")
    
    # Si no se cargó archivo, usar datos de ejemplo
    if df is None:
        st.warning("No se encontraron archivos de datos; usando ejemplo.")
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
        df = pd.DataFrame(data_dict)
    
    # MAPEO DE COLUMNAS: nombres exactos del Excel → nombres esperados en el código
    mapeo_columnas = {
        'Folio': 'Folio',
        'Edad': 'Edad',
        'Sexo': 'Sexo',
        'Altura (en cm)': 'Altura (en cm)',
        'Peso (en kg)': 'Peso (en kg)',
        '¿En los últimos 3 días has dormido al menos 7 horas de forma regular?': 'Sueño_Col',
        '¿Cómo te sientes después del período de recuperación?': 'Recuperacion_Col',
        'Temperatura ambiente (°C)': 'Temp_Ambiente',
        'Humedad Realtiva (%)': 'Humedad',
    }
    
    # Aplicar mapeo: si existe la columna original, renombrarla
    rename_dict = {}
    for col_original, col_nuevo in mapeo_columnas.items():
        if col_original in df.columns:
            rename_dict[col_original] = col_nuevo
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # Rellenar valores faltantes de Edad y Sexo
    if 'Edad' not in df.columns:
        df['Edad'] = 0
    else:
        df['Edad'] = pd.to_numeric(df['Edad'], errors='coerce').fillna(0).astype(int)
    
    if 'Sexo' not in df.columns:
        df['Sexo'] = 'Desconocido'
    else:
        df['Sexo'] = df['Sexo'].fillna('Desconocido')
    
    if 'Altura (en cm)' not in df.columns:
        df['Altura (en cm)'] = 0
    else:
        df['Altura (en cm)'] = pd.to_numeric(df['Altura (en cm)'], errors='coerce').fillna(0)
    
    if 'Peso (en kg)' not in df.columns:
        df['Peso (en kg)'] = 0
    else:
        df['Peso (en kg)'] = pd.to_numeric(df['Peso (en kg)'], errors='coerce').fillna(0)
    
    if 'Temp_Ambiente' not in df.columns:
        df['Temp_Ambiente'] = np.nan
    else:
        df['Temp_Ambiente'] = pd.to_numeric(df['Temp_Ambiente'], errors='coerce')
    
    if 'Humedad' not in df.columns:
        df['Humedad'] = np.nan
    else:
        df['Humedad'] = pd.to_numeric(df['Humedad'], errors='coerce')
    
    # Feature Engineering
    # IMC
    df['IMC'] = df['Peso (en kg)'] / ((df['Altura (en cm)']/100) ** 2)
    df['IMC'] = df['IMC'].replace([np.inf, -np.inf], np.nan)
    
    # Sexo_Num
    le_sexo = LabelEncoder()
    df['Sexo_Num'] = le_sexo.fit_transform(df['Sexo'])
    
    # Sueño_Bien
    if 'Sueño_Col' in df.columns:
        df['Sueño_Bien'] = df['Sueño_Col'].apply(lambda x: 1 if str(x).strip().lower() in ['si','sí','s'] else 0)
    else:
        df['Sueño_Bien'] = 0
    
    # Deltas (si no existen columnas térmicas específicas, crear NaN)
    df['Delta_Ejercicio'] = np.nan
    df['Delta_Recuperacion'] = np.nan
    
    # Mapeo de fatiga
    mapa_fatiga = {
        'Muy recuperado(a), con energía para entrenar.': 0,
        'Recuperado(a), pero aún con algo de fatiga.': 1,
        'Neutral, ni muy descansado(a) ni muy cansado(a).': 1,
        'Algo fatigado(a), me falta recuperar un poco más.': 2,
        'Me siento triste y sin energía.': 2
    }
    
    if 'Recuperacion_Col' in df.columns:
        df['Nivel_Fatiga'] = df['Recuperacion_Col'].map(mapa_fatiga).fillna(1)
    else:
        df['Nivel_Fatiga'] = 1
    
    # Limpiar columnas temporales
    df = df.drop(columns=['Sueño_Col', 'Recuperacion_Col'], errors='ignore')
    
    return df


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
        # Verificar que las columnas de temperatura máxima existan antes de graficar
        required_cols = ['T_Max_Basal', 'T_Max_Post', 'T_Max_Rec']
        if all(c in df_final.columns for c in required_cols):
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
        else:
            st.warning('No hay columnas de temperaturas máximas (T_Max_Basal/T_Max_Post/T_Max_Rec) en los datos; omitiendo gráfico.')
    
    with col2:
        st.subheader("Delta Térmico (Cambios de Temperatura)")
        # Verificar que existan las columnas de delta antes de graficar
        if 'Delta_Ejercicio' in df_final.columns or 'Delta_Recuperacion' in df_final.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(df_final))
            width = 0.35
            # Usar columnas si existen, en otro caso crear serie de NaN
            de = df_final['Delta_Ejercicio'] if 'Delta_Ejercicio' in df_final.columns else np.nan
            dr = df_final['Delta_Recuperacion'] if 'Delta_Recuperacion' in df_final.columns else np.nan
            ax.bar(x - width/2, de, width, label='Delta Ejercicio', color='#f39c12')
            ax.bar(x + width/2, dr, width, label='Delta Recuperación', color='#9b59b6')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Folio')
            ax.set_ylabel('Cambio de Temperatura (°C)')
            ax.set_title('Deltas Térmicos por Atleta')
            ax.set_xticks(x)
            ax.set_xticklabels(df_final['Folio'])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
        else:
            st.warning('No hay columnas de Delta térmico en los datos; omitiendo gráfico.')
    
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
    
    # Entrenar modelo (usar sólo las características disponibles)
    candidate_features = [
        'Edad', 'Sexo_Num', 'IMC', 'Sueño_Bien',
        'T_Max_Basal', 'Delta_Ejercicio', 'Delta_Recuperacion',
        'Temp_Ambiente', 'Humedad'
    ]
    features = [f for f in candidate_features if f in df_final.columns]

    trained = False
    if len(features) == 0:
        st.warning('No hay características disponibles para entrenar el modelo.')
    else:
        X = df_final[features]
        y = df_final['Nivel_Fatiga'] if 'Nivel_Fatiga' in df_final.columns else None

        if y is None or len(y.unique()) < 2:
            st.warning('No hay suficientes etiquetas para entrenar el modelo (se requieren >=2 clases).')
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            trained = True

    # Importancia de variables
    col1, col2 = st.columns(2)

    if trained:
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
    else:
        with col1:
            st.info('Modelo no entrenado — no hay suficientes datos o características.')
        with col2:
            st.info('Modelo no entrenado — no hay suficientes datos o características.')
    
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
