import sqlite3
import pandas as pd
import os

# --- CONFIGURACIÓN ---
# Cambia estos valores para que coincidan con tus archivos y carpetas
DB_FILE = 'ciclistas.db'
CUESTIONARIOS_CSV = 'cuestionarios.csv'
IMAGENES_ROOT_DIR = 'data'


def crear_base_de_datos(conn):
    """Crea las tablas de la base de datos si no existen."""
    cursor = conn.cursor()
    
    # Tabla de Sujetos
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Sujetos (
            id_sujeto INTEGER PRIMARY KEY,
            nombre_sujeto TEXT NOT NULL,
            edad INTEGER,
            sexo TEXT,
            altura_cm REAL,
            peso_kg REAL,
            horas_ejercicio_semana REAL,
            duerme_7_horas BOOLEAN,
            percepcion_esfuerzo INTEGER,
            dolor_pedaleo BOOLEAN,
            conf_no_alcohol_nicotina BOOLEAN,
            conf_no_cosmeticos BOOLEAN,
            conf_hidratado BOOLEAN
        );
    ''')
    
    # Tabla de Imágenes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Imagenes (
            id_imagen INTEGER PRIMARY KEY AUTOINCREMENT,
            id_sujeto INTEGER,
            etapa TEXT NOT NULL,
            ruta_archivo TEXT NOT NULL UNIQUE,
            FOREIGN KEY (id_sujeto) REFERENCES Sujetos (id_sujeto)
        );
    ''')
    
    # Tabla de Métricas (se crea vacía)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Metricas (
            id_metrica INTEGER PRIMARY KEY AUTOINCREMENT,
            id_imagen INTEGER,
            roi_musculo TEXT NOT NULL,
            T_promedio REAL,
            T_max REAL,
            T_min REAL,
            desv_est_T REAL,
            FOREIGN KEY (id_imagen) REFERENCES Imagenes (id_imagen)
        );
    ''')
    
    conn.commit()
    print("✅ Tablas creadas o ya existentes.")


def poblar_tabla_sujetos(conn, archivo_csv):
    """Lee el archivo CSV y lo inserta en la tabla Sujetos."""
    try:
        df_sujetos = pd.read_csv(archivo_csv)
        # Asegúrate que los nombres de las columnas en tu CSV coincidan
        # con los de la tabla 'Sujetos'.
        df_sujetos.to_sql('Sujetos', conn, if_exists='replace', index=False,
                         # Definir el tipo de dato para la clave primaria
                         dtype={'id_sujeto': 'INTEGER PRIMARY KEY'})
        print(f"✅ Tabla 'Sujetos' poblada con {len(df_sujetos)} registros.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{archivo_csv}'.")
    except Exception as e:
        print(f"❌ Ocurrió un error al poblar la tabla Sujetos: {e}")

def poblar_tabla_imagenes(conn, directorio_imagenes):
    """Escanea el directorio de imágenes y puebla la tabla Imagenes."""
    cursor = conn.cursor()
    imagenes_insertadas = 0
    
    if not os.path.isdir(directorio_imagenes):
        print(f"❌ Error: El directorio de imágenes '{directorio_imagenes}' no existe.")
        return

    # Recorrer las carpetas de cada sujeto (ej. 'sujeto_1', 'sujeto_2')
    for nombre_carpeta_sujeto in os.listdir(directorio_imagenes):
        # Extraer el ID del sujeto del nombre de la carpeta
        if nombre_carpeta_sujeto.startswith('sujeto_'):
            try:
                id_sujeto = int(nombre_carpeta_sujeto.split('_')[1])
                ruta_carpeta_sujeto = os.path.join(directorio_imagenes, nombre_carpeta_sujeto)

                # Recorrer cada archivo de imagen dentro de la carpeta del sujeto
                for nombre_archivo in os.listdir(ruta_carpeta_sujeto):
                    etapa = os.path.splitext(nombre_archivo)[0] # Ej: 'pre_ejercicio'
                    ruta_completa = os.path.join(ruta_carpeta_sujeto, nombre_archivo)
                    
                    # Insertar la información si no existe ya (evita duplicados)
                    cursor.execute(
                        "INSERT OR IGNORE INTO Imagenes (id_sujeto, etapa, ruta_archivo) VALUES (?, ?, ?)",
                        (id_sujeto, etapa, ruta_completa)
                    )
                    if cursor.rowcount > 0:
                        imagenes_insertadas += 1

            except (ValueError, IndexError):
                print(f"⚠️  Advertencia: No se pudo procesar la carpeta '{nombre_carpeta_sujeto}'.")
    
    conn.commit()
    print(f"✅ Tabla 'Imagenes' actualizada. Se insertaron {imagenes_insertadas} nuevas imágenes.")


# --- Ejecución Principal ---
if __name__ == "__main__":
    print("🚀 Iniciando script de configuración de la base de datos...")
    
    # Crear conexión con la base de datos
    connection = sqlite3.connect(DB_FILE)
    
    # 1. Crear la estructura de tablas
    crear_base_de_datos(connection)
    
    # 2. Poblar la tabla de sujetos desde el CSV
    poblar_tabla_sujetos(connection, CUESTIONARIOS_CSV)
    
    # 3. Escanear carpetas y poblar la tabla de imágenes
    poblar_tabla_imagenes(connection, IMAGENES_ROOT_DIR)
    
    # Cerrar la conexión
    connection.close()
    
    print("\n🎉 Proceso finalizado. Tu base de datos está lista.")