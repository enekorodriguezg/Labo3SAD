import sys
import subprocess
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def main():
    if len(sys.argv) < 5:
        print("Uso: python barrido.py <archivo.csv> <k_minima> <k_maxima> <p_maxima>")
        sys.exit(1)

    archivo_datos = sys.argv[1]
    k_min = int(sys.argv[2])
    k_max = int(sys.argv[3])
    p_max = int(sys.argv[4])
    archivo_config = "configuration.json"
    pesos = ['uniform', 'distance']

    print(f"\n--- PASO 1: Dividiendo {archivo_datos} en 80% (Entrenamiento) y 20% (Test Ciego) ---")
    df = pd.read_csv(archivo_datos)

    # Dividimos los datos
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Guardamos el 80% para dárselo a train.py
    archivo_train = "datos_entrenamiento_80.csv"
    df_train.to_csv(archivo_train, index=False)

    # Guardamos el 20% para test.py (le quitamos la última columna, que es la solución, para que prediga a ciegas)
    archivo_test = "datos_test_20_ciego.csv"
    df_test_sin_clase = df_test.iloc[:, :-1]
    df_test_sin_clase.to_csv(archivo_test, index=False)

    # Limpiamos resultados anteriores para que no se mezclen
    if os.path.exists('resultados_knn.csv'):
        os.remove('resultados_knn.csv')

    print("\n--- PASO 2: Entrenando todas las combinaciones posibles ---")
    for k in range(k_min, k_max + 1, 2):
        for p in range(1, p_max + 1):
            for w in pesos:
                comando = [
                    "python", "train.py",
                    archivo_train, str(k), w, str(p),
                    "-c", archivo_config
                ]
                subprocess.run(comando)

    print("\n--- PASO 3: Buscando el mejor modelo... ---")
    resultados = pd.read_csv('resultados_knn.csv')
    # Buscamos la fila con el mejor F-score
    mejor_fila = resultados.loc[resultados['F_score_macro'].idxmax()]

    # Extraemos los datos ganadores de la columna 'Combinacion'
    partes = mejor_fila['Combinacion'].split(' ')
    mejor_k = partes[0].split('=')[1]
    mejor_p = partes[1].split('=')[1]
    mejor_w = partes[2]

    print(f"¡Ganador! k={mejor_k}, p={mejor_p}, pesos={mejor_w} (F-score: {mejor_fila['F_score_macro']:.4f})")

    print("\n--- PASO 4: Guardando ÚNICAMENTE el modelo ganador ---")
    # Llamamos a train.py pasándole un nuevo parámetro: --save
    comando_guardar = [
        "python", "train.py",
        archivo_train, mejor_k, mejor_w, mejor_p,
        "-c", archivo_config,
        "--save"
    ]
    subprocess.run(comando_guardar)

    nombre_modelo = f"modelo_k{mejor_k}_p{mejor_p}_{mejor_w}.sav"

    print("\n--- PASO 5: Pasando el 20% ciego por el modelo ganador (test.py) ---")
    comando_test = [
        "python", "test.py",
        archivo_test, nombre_modelo
    ]
    subprocess.run(comando_test)


if __name__ == "__main__":
    main()