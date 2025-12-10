import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# 1. CARGA Y LIMPIEZA DE DATOS (Encuesta)
# =============================================================================

# Nombre de tu archivo Excel (Asegúrate de que esté en la misma carpeta que este script)
archivo_excel = "Monitoreo_de_Fatiga.xlsx" 

# Simulación de carga (Descomenta la siguiente línea si tienes el archivo real listo)
# df_encuesta = pd.read_excel(archivo_excel)

# --- CREACIÓN DE DATOS DE EJEMPLO BASADOS EN TU TABLA (FUENTE [1]) ---
# (Esto es para que el código funcione demostrativamente ahora mismo)
data_dict = {
    'Folio': ['F-001', 'F-002', 'F-003', 'F-004', 'F-005', 'F-006'],
    'Edad': [22, 25, 19, 28, 21, 24],
    'Sexo': ['Masculino', 'Femenino', 'Masculino', 'Femenino', 'Femenino', 'Masculino'],
    'Altura (en cm)': [175, 162, 180, 158, 165, 172],
    'Peso (en kg)': [75, 58, 82, 55, 62, 80],
    '¿En los últimos 3 días has dormido al menos 7 horas de forma regular?': ['Si', 'No', 'Si', 'No', 'No', 'Si'],
    # Target (Variable Objetivo)
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
print("--- Datos de Encuesta Cargados (Ejemplo) ---")

# =============================================================================
# 2. INTEGRACIÓN DE DATOS TÉRMICOS (Imágenes)
# =============================================================================
# NOTA: Como tus temperaturas están en las imágenes JPG (fuentes 1-24), 
# necesitas un archivo intermedio (Excel o CSV) que tenga las temperaturas extraídas.
# Aquí simulamos esos datos basándonos en tus imágenes F-001.

# T1 = Basal, TF/T3 = Post-Ejercicio, T5 = Recuperación
datos_termicos = pd.DataFrame({
    'Folio': ['F-001', 'F-002', 'F-003', 'F-004', 'F-005', 'F-006'],
    # Temperaturas Máximas del Cuádriceps (Ejemplos aproximados)
    'T_Max_Basal': [34.4, 33.5, 34.0, 33.8, 34.2, 33.9],  # T1
    'T_Max_Post':  [34.2, 35.5, 33.8, 35.0, 35.2, 36.1],  # TF/T3
    'T_Max_Rec':   [35.3, 34.8, 34.5, 34.5, 34.9, 35.8],  # T5
    # Datos ambientales (CRÍTICOS según fuente [3])
    'Temp_Ambiente': [11.5, 12.0, 11.8, 12.5, 11.2, 12.1],
    'Humedad': [14.5, 15.0, 16.2, 14.8, 15.5, 16.0]
})

# Unir ambas tablas usando el Folio
df_final = pd.merge(df_encuesta, datos_termicos, on='Folio')

# =============================================================================
# 3. INGENIERÍA DE CARACTERÍSTICAS (Feature Engineering)
# =============================================================================

# A. Calcular IMC (Peso / Altura^2)
df_final['IMC'] = df_final['Peso (en kg)'] / ((df_final['Altura (en cm)']/100) ** 2)

# B. Codificar Variables Categóricas
le_sexo = LabelEncoder()
df_final['Sexo_Num'] = le_sexo.fit_transform(df_final['Sexo']) # M=1, F=0 (aprox)

df_final['Sueño_Bien'] = df_final['¿En los últimos 3 días has dormido al menos 7 horas de forma regular?'].apply(
    lambda x: 1 if x == 'Si' else 0
)

# C. CÁLCULO DE DELTAS TÉRMICOS (Clave del algoritmo)
# Delta Ejercicio: Diferencia entre Post y Basal. 
# Teoría: Debería bajar o mantenerse por vasoconstricción. Si sube mucho -> ineficiencia/fatiga.
df_final['Delta_Ejercicio'] = df_final['T_Max_Post'] - df_final['T_Max_Basal']

# Delta Recuperación: Diferencia entre Recuperación y Post.
# Teoría: Capacidad de disipar calor.
df_final['Delta_Recuperacion'] = df_final['T_Max_Rec'] - df_final['T_Max_Post']

# D. Codificar la Variable Objetivo (Lo que queremos predecir)
target_col = '¿Cómo te sientes después del período de recuperación?'

# Mapeo manual para asegurar orden de fatiga (0 = Bien, 2 = Fatigado)
# Ajusta esto según las respuestas exactas de tu Excel
mapa_fatiga = {
    'Muy recuperado(a), con energía para entrenar.': 0,
    'Recuperado(a), pero aún con algo de fatiga.': 1,
    'Neutral, ni muy descansado(a) ni muy cansado(a).': 1, # Agrupamos neutral con fatiga leve
    'Algo fatigado(a), me falta recuperar un poco más.': 2,
    'Me siento triste y sin energía.': 2
}

# Aplicar mapeo (rellena con 1 si encuentra texto nuevo)
df_final['Nivel_Fatiga'] = df_final[target_col].map(mapa_fatiga).fillna(1)

# =============================================================================
# 4. ENTRENAMIENTO DEL MODELO
# =============================================================================

# Definir las características que usará el "cerebro" del modelo
features = [
    'Edad', 'Sexo_Num', 'IMC', 'Sueño_Bien',
    'T_Max_Basal', 'Delta_Ejercicio', 'Delta_Recuperacion',
    'Temp_Ambiente', 'Humedad'
]

X = df_final[features]
y = df_final['Nivel_Fatiga']

# Dividir en entrenamiento (Train) y prueba (Test)
# Nota: Con pocos datos (6 filas en el ejemplo), el test será muy pequeño. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el Modelo (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar
print("\nEntrenando modelo...")
model.fit(X_train, y_train)

# =============================================================================
# 5. RESULTADOS Y ANÁLISIS
# =============================================================================

# Predicción en el set de prueba
y_pred = model.predict(X_test)

print("\n--- RESULTADOS DEL MODELO ---")
print("Clases predichas:", y_pred)
print("Clases reales:   ", y_test.values)

# Importancia de las variables: ¿Qué influyó más en la fatiga?
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n--- FACTORES MÁS IMPORTANTES PARA PREDECIR FATIGA ---")
print(importances)

print("\n--- EXPLICACIÓN ---")
top_factor = importances.index
print(f"El modelo sugiere que '{top_factor}' es el indicador más fuerte en este set de datos.")
print("Si 'Delta_Ejercicio' es alto, significa que la respuesta térmica durante el esfuerzo es clave.")
print("Si 'Sueño_Bien' es alto, los hábitos de descanso predominan sobre la termografía.")

# Guardar el modelo para uso futuro
# import joblib
# joblib.dump(model, 'modelo_fatiga_ciclistas.pkl')