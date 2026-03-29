import sys
import pandas as pd
import pickle
import os

def main():
    if len(sys.argv) < 3: #Permite ejecutar el programa desde la terminal pasándole los archivos
        print("Uso: python test.py <datos_nuevos.csv> <modelo_guardado.sav>")
        sys.exit(1)

    archivo_nuevos_datos = sys.argv[1]
    archivo_modelo = sys.argv[2]

    print(f"Cargando el modelo {archivo_modelo}...")
    try:
        clf = pickle.load(open(archivo_modelo, 'rb')) #Se abre el archivo .sav y se "despierta" al modelo, guadándolo en la variable clf
    except FileNotFoundError:
        print(f"Error: No se encuentra el modelo {archivo_modelo}.")
        sys.exit(1)

    # Cargar obligatoriamente el LabelEncoder para traducir los números de vuelta a texto.
    try:
        le = pickle.load(open('label_encoder.sav', 'rb'))
    except FileNotFoundError:
        print("Error crítico: No se encuentra 'label_encoder.sav'. Es necesario para traducir las predicciones.")
        sys.exit(1)

    print(f"Cargando nuevas instancias desde {archivo_nuevos_datos}...")
    try:
        X_nuevo = pd.read_csv(archivo_nuevos_datos)
    except FileNotFoundError:
        print(f"Error: No se encuentra el archivo de datos {archivo_nuevos_datos}.")
        sys.exit(1)

    try:
        #Se busca si el archivo nuevo trae una columna de "ID". Si la tiene, se quita temporalmente, porque si se le pasa el ID al modelo para predecir, el modelo se rompe, porque no ha entrenado con esa columna. Se guarda en columna_id_datos
        columna_id_nombre = None
        columna_id_datos = None
        posibles_nombres_id = ['id', 'id_cliente', 'identifier', 'identificador', 'passengerid', 'customerid']
        
        for col in X_nuevo.columns:
            if col.lower() in posibles_nombres_id:
                columna_id_nombre = col
                columna_id_datos = X_nuevo[col].copy()
                X_nuevo = X_nuevo.drop(columns=[col])
                print(f"[INFO] Columna '{col}' detectada como identificador y separada temporalmente.")
                break # Solo extraemos la primera que coincida

        # Blindaje contra columnas sobrantes (Ej: Si el dataset tiene la columna objetivo incluida)
        if hasattr(clf, 'feature_names_in_'):
            columnas_esperadas = list(clf.feature_names_in_)
            columnas_sobrantes = set(X_nuevo.columns) - set(columnas_esperadas)
            if columnas_sobrantes:
                X_nuevo = X_nuevo.drop(columns=list(columnas_sobrantes))
                print(f"[INFO] Columnas sobrantes extirpadas para evitar fallos de dimensionalidad: {columnas_sobrantes}")

        resultado_numerico = clf.predict(X_nuevo) #El modelo coge los datos nuevos, los pasa por el imputador (si faltan datos), los escala, y escupe sus predicciones numéricas

        # Restaurar el ID original si existía
        if columna_id_nombre is not None:
            X_nuevo.insert(0, columna_id_nombre, columna_id_datos)

        X_nuevo = X_nuevo.copy()

        # Inyección de las predicciones traducidas (Texto, no números)
        X_nuevo['Prediccion_Clase'] = le.inverse_transform(resultado_numerico) #Se usa el traductor para convertir esos números fríos en las palabras reales y se guardan en una columna nueva llamada Prediccion_Clase

        nombre_puro = os.path.basename(archivo_modelo)
        nombre_base_modelo = nombre_puro.replace('.sav', '')
        archivo_salida = f"predicciones_{nombre_base_modelo}.csv"

        X_nuevo.to_csv(archivo_salida, index=False)
        print(f"\n¡Clasificación completada! Resultados legibles en: {archivo_salida}")

    except Exception as e:
        print(f"\nError crítico al predecir: {e}")

if __name__ == "__main__":
    main()