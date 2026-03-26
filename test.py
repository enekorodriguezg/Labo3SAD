import sys
import pandas as pd
import pickle


def main():
    if len(sys.argv) < 3:
        print("Uso: python test.py <datos_nuevos.csv> <modelo_guardado.sav>")
        sys.exit(1)

    archivo_nuevos_datos = sys.argv[1]
    archivo_modelo = sys.argv[2]

    print(f"Cargando el modelo {archivo_modelo}...")
    try:
        clf = pickle.load(open(archivo_modelo, 'rb'))
    except FileNotFoundError:
        print(f"Error: No se encuentra el modelo {archivo_modelo}.")
        sys.exit(1)

    print(f"Cargando nuevas instancias desde {archivo_nuevos_datos}...")
    try:
        X_nuevo = pd.read_csv(archivo_nuevos_datos)
    except FileNotFoundError:
        print(f"Error: No se encuentra el archivo de datos {archivo_nuevos_datos}.")
        sys.exit(1)

    try:
        # CIRUGÍA VITAL: Secuestrar el ID temporalmente
        columna_id = None
        if 'ID' in X_nuevo.columns:
            columna_id = X_nuevo['ID'].copy()
            X_nuevo = X_nuevo.drop(columns=['ID'])
            print("[INFO] Columna 'ID' separada temporalmente para la predicción.")

        # Inferencia matemática pura
        resultado = clf.predict(X_nuevo)

        # Restaurar el ID al principio del dataframe para la entrega
        if columna_id is not None:
            X_nuevo.insert(0, 'ID', columna_id)

        # CIRUGÍA: Desfragmentar la memoria RAM antes de añadir la última columna
        X_nuevo = X_nuevo.copy()

        # Añadir la columna con la solución
        X_nuevo['Prediccion_Clase'] = resultado

        # BLINDAJE: Nombramos el CSV de salida dinámicamente según el modelo usado
        nombre_base_modelo = archivo_modelo.replace('.sav', '')
        archivo_salida = f"predicciones_{nombre_base_modelo}.csv"
        X_nuevo.to_csv(archivo_salida, index=False)

        print(f"\n¡Clasificación completada! Resultados en: {archivo_salida}")

    except Exception as e:
        print(f"\nError crítico al predecir: {e}")


if __name__ == "__main__":
    main()