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

    # 1. Recargar el modelo del disco
    try:
        clf = pickle.load(open(archivo_modelo, 'rb'))
    except FileNotFoundError:
        print(f"Error: No se encuentra el modelo {archivo_modelo}.")
        sys.exit(1)

    print(f"Cargando nuevas instancias desde {archivo_nuevos_datos}...")

    # 2. Cargar los datos nuevos (OJO: estos datos NO deben tener la columna 'Especie' o target)
    try:
        X_nuevo = pd.read_csv(archivo_nuevos_datos)
    except FileNotFoundError:
        print(f"Error: No se encuentra el archivo de datos {archivo_nuevos_datos}.")
        sys.exit(1)

    # 3. Clasificar las nuevas instancias
    try:
        resultado = clf.predict(X_nuevo)

        # Opcional: Añadir las predicciones como una nueva columna al dataframe original
        X_nuevo['Prediccion_Clase'] = resultado

        # Guardar el resultado en un nuevo CSV para que puedas entregarlo
        archivo_salida = "predicciones_finales.csv"
        X_nuevo.to_csv(archivo_salida, index=False)

        print("\n¡Clasificación completada!")
        print(f"Los resultados se han guardado en: {archivo_salida}")
        print(X_nuevo.head())  # Mostramos las 5 primeras predicciones

    except Exception as e:
        print(f"\nError al predecir: {e}")
        print(
            "Asegúrate de que el CSV de datos nuevos tiene exactamente las mismas columnas (y en el mismo orden) que los datos de entrenamiento, SIN la columna objetivo.")


if __name__ == "__main__":
    main()