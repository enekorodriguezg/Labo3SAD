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
    parser = argparse.ArgumentParser(description='Ejecutar kNN con configuración dinámica.')
    parser.add_argument('file', type=str, help='Ruta al dataset (ej. iris.csv)')
    parser.add_argument('k', type=int, help='Número de vecinos (k)')
    parser.add_argument('weights', type=str, help='Pesos: uniform o distance')
    parser.add_argument('p', type=int, help='Parámetro p de distancia (1 o 2)')
    parser.add_argument('-c', '--config', type=str, required=True, help='Archivo JSON de configuración')

    # NUEVO: Argumento opcional para decirle cuándo debe guardar el modelo
    parser.add_argument('--save', action='store_true', help='Si se incluye, guarda el modelo .sav')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)["preprocessing"]

    df = pd.read_csv(args.file)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le = LabelEncoder()
    y = le.fit_transform(y)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=config.get('impute_strategy', 'mean'))),
        ('scaler', StandardScaler() if config.get('scaling') == 'standard' else 'passthrough')
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ], remainder='passthrough')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=args.k, weights=args.weights, p=args.p))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    resultado_str = f"k={args.k} p={args.p} {args.weights},{prec:.4f},{rec:.4f},{f1:.4f},{acc:.4f}\n"

    csv_file = 'resultados_knn.csv'
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a') as f:
        if not file_exists:
            f.write("Combinacion,Precision,Recall,F_score_macro,Accuracy\n")
        f.write(resultado_str)

    # NUEVO: Solo guarda el disco si le hemos pasado el flag --save desde barrido.py
    if args.save:
        # Volvemos a entrenar, pero esta vez con todo el 80% junto para que
        # el modelo final sea lo más inteligente posible antes de guardarlo.
        clf.fit(X, y)
        model_name = f"modelo_k{args.k}_p{args.p}_{args.weights}.sav"
        pickle.dump(clf, open(model_name, 'wb'))
        print(f" ---> ¡Modelo definitivo guardado como {model_name}!")


if __name__ == "__main__":
    main()