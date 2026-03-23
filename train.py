import sys
import json
import argparse
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    # 1. Configurar los argumentos de entrada
    parser = argparse.ArgumentParser(description='Ejecutar kNN con configuración dinámica.')
    parser.add_argument('file', type=str, help='Ruta al dataset (ej. iris.csv)')
    parser.add_argument('k', type=int, help='Número de vecinos (k)')
    parser.add_argument('weights', type=str, help='Pesos: uniform o distance')
    parser.add_argument('p', type=int, help='Parámetro p de distancia (1 o 2)')
    parser.add_argument('-c', '--config', type=str, required=True, help='Archivo JSON de configuración')

    args = parser.parse_args()

    # 2. Leer la configuración del JSON
    with open(args.config, 'r') as f:
        config = json.load(f)["preprocessing"]

    # 3. Cargar los datos
    df = pd.read_csv(args.file)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Codificar la clase objetivo a números (ej. Iris-setosa -> 0)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 4. Preproceso Dinámico usando Pipeline
    # Seleccionamos solo las columnas numéricas para imputar y escalar
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=config.get('impute_strategy', 'mean'))),
        ('scaler', StandardScaler() if config.get('scaling') == 'standard' else 'passthrough')
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ], remainder='passthrough')

    # 5. División de datos y definición del modelo completo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=args.k, weights=args.weights, p=args.p))
    ])

    # Entrenar
    clf.fit(X_train, y_train)

    # Predecir
    y_pred = clf.predict(X_test)

    # 6. Calcular Figuras de Mérito
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # 7. Generar CSV de métricas
    resultado_str = f"k={args.k} p={args.p} {args.weights},{prec:.4f},{rec:.4f},{f1:.4f},{acc:.4f}\n"

    csv_file = 'resultados_knn.csv'
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a') as f:
        if not file_exists:
            f.write("Combinacion,Precision,Recall,F_score_macro,Accuracy\n")
        f.write(resultado_str)

    # 8. Guardar el modelo en disco con pickle
    model_name = f"modelo_k{args.k}_p{args.p}_{args.weights}.sav"
    pickle.dump(clf, open(model_name, 'wb'))

    print(
        f"Completado: k={args.k}, p={args.p}, w={args.weights} -> F-score: {f1:.4f} | Modelo guardado como {model_name}")


if __name__ == "__main__":
    main()