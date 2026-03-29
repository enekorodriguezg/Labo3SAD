import argparse
import sys
import json
import pandas as pd
import pickle
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

def guardar_metricas(cv_results, nombre_archivo):
    df_res = pd.DataFrame(cv_results)
    cols_to_keep = ['params', 'mean_test_precision', 'mean_test_recall', 'mean_test_f1_macro', 'mean_test_accuracy']
    df_clean = df_res[cols_to_keep].rename(columns={
        'params': 'Combinacion',
        'mean_test_precision': 'Precision',
        'mean_test_recall': 'Recall',
        'mean_test_f1_macro': 'F_score_macro',
        'mean_test_accuracy': 'Accuracy'
    })
    df_clean.to_csv(nombre_archivo, index=False)
    print(f"Métricas exportadas a: {nombre_archivo}")

def main():
    parser = argparse.ArgumentParser(description='Entrenamiento Universal de modelos ML')
    parser.add_argument('archivo_datos', type=str, help='Ruta al dataset')
    parser.add_argument('--algo', type=str, choices=['knn', 'tree', 'nb', 'rf', 'all'], default='all')
    parser.add_argument('-c', '--config', type=str, required=True, help='Ruta al archivo JSON de configuración')

    args = parser.parse_args()
    archivo_config = args.config

    print(f"1. Ingesta de Datos y Limpieza")
    try:
        df = pd.read_csv(args.archivo_datos)
    except FileNotFoundError:
        print(f"Error crítico: No se encuentra {args.archivo_datos}")
        sys.exit(1)

    # Búsqueda dinámica de la columna ID. No dependemos de mayúsculas ni nombres exactos.
    posibles_nombres_id = ['id', 'id_cliente', 'identifier', 'identificador', 'passengerid', 'customerid']
    columnas_a_eliminar = [col for col in df.columns if col.lower() in posibles_nombres_id]
    
    if columnas_a_eliminar:
        df = df.drop(columns=columnas_a_eliminar)
        print(f"Columna(s) identificadora(s) detectada(s) y eliminada(s): {columnas_a_eliminar}")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Guardamos el traductor de clases para que test.py sepa qué significa 0, 1, 2...
    pickle.dump(le, open('label_encoder.sav', 'wb'))
    print("LabelEncoder guardado dinámicamente en 'label_encoder.sav'.")

    print(f"2. Partición: 80% Entrenamiento / 20% Test Ciego")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    archivo_test = "datos_test_20_ciego.csv"
    X_test.to_csv(archivo_test, index=False)

    archivo_soluciones = "datos_test_20_soluciones.csv"
    pd.DataFrame({'Etiqueta_Real': le.inverse_transform(y_test)}).to_csv(archivo_soluciones, index=False)

    print("3. Construyendo Preprocesador Dinámico")
    with open(archivo_config, 'r') as f:
        config_completo = json.load(f)
        config = config_completo["preprocessing"]

    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    numeric_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy=config.get('impute_strategy', 'median'))),
        ('scaler', StandardScaler() if config.get('scaling') == 'standard' else 'passthrough')
    ])

    categorical_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough', sparse_threshold=0)

    print("4. Ensamblando Pipelines con Balanceo de Clases")
    tipo_muestreo = config.get('sampling', 'none')
    sampler_step = None
    if tipo_muestreo == 'undersampling':
        sampler_step = ('sampler', RandomUnderSampler(random_state=42))
        print("Muestreo: Undersampling activado.")
    elif tipo_muestreo == 'oversampling':
        sampler_step = ('sampler', SMOTE(random_state=42))
        print("Muestreo: Oversampling (SMOTE) activado.")

    pasos_knn = [('preprocessor', preprocessor)]
    pasos_tree = [('preprocessor', preprocessor)]
    pasos_nb = [('preprocessor', preprocessor)]
    pasos_rf = [('preprocessor', preprocessor)]

    if sampler_step:
        for pasos in [pasos_knn, pasos_tree, pasos_nb, pasos_rf]:
            pasos.append(sampler_step)

    pasos_knn.append(('classifier', KNeighborsClassifier()))
    pasos_tree.append(('classifier', DecisionTreeClassifier(random_state=42)))
    pasos_nb.append(('classifier', GaussianNB()))
    pasos_rf.append(('classifier', RandomForestClassifier(random_state=42, n_jobs=-1)))

    pipe_knn = ImbPipeline(steps=pasos_knn)
    pipe_tree = ImbPipeline(steps=pasos_tree)
    pipe_nb = ImbPipeline(steps=pasos_nb)
    pipe_rf = ImbPipeline(steps=pasos_rf)

    print("5. Definiendo Espacios de Búsqueda")
    hyperparams = config_completo.get("hyperparameters", {})

    metricas = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1_macro': 'f1_macro'
    }

    print("6. Ejecutando Entrenamiento y Validación Cruzada")
    modelos_ganadores = []
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics')

    # Ejecución compactada para legibilidad
    algoritmos = {
        'knn': (pipe_knn, hyperparams.get("knn", {})),
        'tree': (pipe_tree, hyperparams.get("tree", {})),
        'nb': (pipe_nb, hyperparams.get("nb", {})),
        'rf': (pipe_rf, hyperparams.get("rf", {}))
    }

    for nombre_algo, (pipeline, parametros) in algoritmos.items():
        if args.algo in [nombre_algo, 'all']:
            print(f"Entrenando {nombre_algo.upper()}...")
            grid = GridSearchCV(pipeline, parametros, cv=5, scoring=metricas, refit='f1_macro', n_jobs=-1)
            grid.fit(X_train, y_train)
            print(f"Ganador {nombre_algo.upper()}: F-score {grid.best_score_:.4f}")
            modelos_ganadores.append((f"mejor_modelo_{nombre_algo}.sav", grid.best_estimator_))
            guardar_metricas(grid.cv_results_, f"resultados_{nombre_algo}.csv")

    print("\n7. Guardando Modelos Físicos")
    for nombre, modelo in modelos_ganadores:
        pickle.dump(modelo, open(nombre, 'wb'))

    print("\nEntrenamiento finalizado con éxito.")

if __name__ == "__main__":
    main()