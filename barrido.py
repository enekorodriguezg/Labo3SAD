import sys
import subprocess


def main():
    # Ahora exigimos 5 elementos (el nombre del script + 4 argumentos)
    if len(sys.argv) < 5:
        print("Uso: python barrido.py <archivo.csv> <k_minima> <k_maxima> <p_maxima>")
        sys.exit(1)

    # Asignamos los argumentos dinámicamente
    archivo_datos = sys.argv[1]
    k_min = int(sys.argv[2])
    k_max = int(sys.argv[3])
    p_max = int(sys.argv[4])

    archivo_config = "configuration.json"
    pesos = ['uniform', 'distance']

    print(f"Iniciando barrido de hiperparámetros sobre {archivo_datos}...")

    for k in range(k_min, k_max + 1, 2):
        for p in range(1, p_max + 1):
            for w in pesos:
                comando = [
                    "python", "train.py",
                    archivo_datos,
                    str(k),
                    w,
                    str(p),
                    "-c", archivo_config
                ]
                # Ejecutamos el subproceso
                subprocess.run(comando)

    print("\nBarrido completado. Revisa el archivo 'resultados_knn.csv' y los modelos .sav generados.")


if __name__ == "__main__":
    main()