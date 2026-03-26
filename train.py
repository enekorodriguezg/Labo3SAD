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

# Algoritmos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Imbalanced-learn (Balanceo seguro en el Pipeline)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


def guardar_metricas(cv_results, nombre_archivo):
    """Extrae las métricas del GridSearchCV y las guarda limpias en un CSV."""
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
    parser.add_argument('--algo', type=str, choices=['knn', 'tree', 'nb', 'rf', 'all'], default='all',
                        help='Algoritmo a ejecutar: knn, tree, nb, rf o all (por defecto)')
    args = parser.parse_args()

    archivo_config = "configuration.json"

    print(f"--- 1. Ingesta de Datos y Limpieza ---")
    try:
        df = pd.read_csv(args.archivo_datos)
    except FileNotFoundError:
        print(f"Error crítico: No se encuentra {args.archivo_datos}")
        sys.exit(1)

    # CIRUGÍA: Extirpamos la columna ID si existe para que no genere ruido matemático
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        print("[INFO] Columna 'ID' detectada y eliminada del entrenamiento.")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le = LabelEncoder()
    y = le.fit_transform(y)

    print(f"--- 2. Partición: 80% Entrenamiento / 20% Test Ciego ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    archivo_test = "datos_test_20_ciego.csv"
    X_test.to_csv(archivo_test, index=False)

    # Guardamos las soluciones para que test.py pueda dibujar la Matriz de Confusión si la pides
    archivo_soluciones = "datos_test_20_soluciones.csv"
    pd.DataFrame({'Etiqueta_Real': y_test}).to_csv(archivo_soluciones, index=False)

    print("--- 3. Construyendo Preprocesador Dinámico (Numérico + Texto) ---")
    with open(archivo_config, 'r') as f:
        config = json.load(f)["preprocessing"]

    # Detectamos automáticamente qué es número y qué es texto
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    # Tubería Numérica
    numeric_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy=config.get('impute_strategy', 'median'))),
        ('scaler', StandardScaler() if config.get('scaling') == 'standard' else 'passthrough')
    ])

    # Tubería Categórica (Texto) -> Vital para Santander
    categorical_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

    print("--- 4. Ensamblando Pipelines con Balanceo de Clases ---")

    tipo_muestreo = config.get('sampling', 'none')
    sampler_step = None
    if tipo_muestreo == 'undersampling':
        sampler_step = ('sampler', RandomUnderSampler(random_state=42))
        print("[INFO] Muestreo: Undersampling activado.")
    elif tipo_muestreo == 'oversampling':
        sampler_step = ('sampler', SMOTE(random_state=42))
        print("[INFO] Muestreo: Oversampling (SMOTE) activado.")

    # Definimos la base de las tuberías
    pasos_knn = [('preprocessor', preprocessor)]
    pasos_tree = [('preprocessor', preprocessor)]
    pasos_nb = [('preprocessor', preprocessor)]
    pasos_rf = [('preprocessor', preprocessor)]

    # Inyectamos el balanceo si el JSON lo pide
    if sampler_step:
        pasos_knn.append(sampler_step)
        pasos_tree.append(sampler_step)
        pasos_nb.append(sampler_step)
        pasos_rf.append(sampler_step)

    pasos_knn.append(('classifier', KNeighborsClassifier()))
    pasos_tree.append(('classifier', DecisionTreeClassifier(random_state=42)))
    pasos_nb.append(('classifier', GaussianNB()))
    pasos_rf.append(('classifier', RandomForestClassifier(random_state=42, n_jobs=-1)))

    pipe_knn = ImbPipeline(steps=pasos_knn)
    pipe_tree = ImbPipeline(steps=pasos_tree)
    pipe_nb = ImbPipeline(steps=pasos_nb)
    pipe_rf = ImbPipeline(steps=pasos_rf)

    print("--- 5. Definiendo Espacios de Búsqueda ---")
    parametros_knn = {
        'classifier__n_neighbors': [3, 5, 7, 9],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__p': [1, 2]
    }

    parametros_tree = {
        'classifier__max_depth': [3, 5, 10, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__criterion': ['gini', 'entropy']
    }

    parametros_nb = {
        'classifier__var_smoothing': [1e-9, 1e-7, 1e-5]
    }

    parametros_rf = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }

    metricas = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1_macro': 'f1_macro'
    }

    print("--- 6. Ejecutando Entrenamiento y Validación Cruzada ---")
    modelos_ganadores = []

    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics')

    if args.algo in ['knn', 'all']:
        print(">>> Entrenando kNN...")
        grid_knn = GridSearchCV(pipe_knn, parametros_knn, cv=5, scoring=metricas, refit='f1_macro', n_jobs=-1)
        grid_knn.fit(X_train, y_train)
        print(f"Ganador kNN: F-score {grid_knn.best_score_:.4f}")
        modelos_ganadores.append(("mejor_modelo_knn.sav", grid_knn.best_estimator_))
        guardar_metricas(grid_knn.cv_results_, "resultados_knn.csv")

    if args.algo in ['tree', 'all']:
        print(">>> Entrenando Árbol de Decisión...")
        grid_tree = GridSearchCV(pipe_tree, parametros_tree, cv=5, scoring=metricas, refit='f1_macro', n_jobs=-1)
        grid_tree.fit(X_train, y_train)
        print(f"Ganador Árbol: F-score {grid_tree.best_score_:.4f}")
        modelos_ganadores.append(("mejor_modelo_tree.sav", grid_tree.best_estimator_))
        guardar_metricas(grid_tree.cv_results_, "resultados_tree.csv")

    if args.algo in ['nb', 'all']:
        print(">>> Entrenando Naive Bayes...")
        grid_nb = GridSearchCV(pipe_nb, parametros_nb, cv=5, scoring=metricas, refit='f1_macro', n_jobs=-1)
        grid_nb.fit(X_train, y_train)
        print(f"Ganador Naive Bayes: F-score {grid_nb.best_score_:.4f}")
        modelos_ganadores.append(("mejor_modelo_nb.sav", grid_nb.best_estimator_))
        guardar_metricas(grid_nb.cv_results_, "resultados_nb.csv")

    if args.algo in ['rf', 'all']:
        print(">>> Entrenando Random Forest...")
        grid_rf = GridSearchCV(pipe_rf, parametros_rf, cv=5, scoring=metricas, refit='f1_macro', n_jobs=-1)
        grid_rf.fit(X_train, y_train)
        print(f"Ganador Random Forest: F-score {grid_rf.best_score_:.4f}")
        modelos_ganadores.append(("mejor_modelo_rf.sav", grid_rf.best_estimator_))
        guardimport argparse
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

# Algoritmos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Imbalanced-learn (Balanceo seguro en el Pipeline)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

def guardar_metricas(cv_results, nombre_archivo):
    """Extrae las métricas del GridSearchCV y las guarda limpias en un CSV."""
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
    parser.add_argument('--algo', type=str, choices=['knn', 'tree', 'nb', 'rf', 'all'], default='all',
                        help='Algoritmo a ejecutar: knn, tree, nb, rf o all (por defecto)')
    parser.add_argument('-c', '--config', type=str, required=True, help='Ruta al archivo JSON de configuración')

    args = parser.parse_args()
    archivo_config = args.config

    print(f"--- 1. Ingesta de Datos y Limpieza ---")
    try:
        df = pd.read_csv(args.archivo_datos)
    except FileNotFoundError:
        print(f"Error crítico: No se encuentra {args.archivo_datos}")
        sys.exit(1)

    # CIRUGÍA: Extirpamos la columna ID si existe para que no genere ruido matemático
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        print("[INFO] Columna 'ID' detectada y eliminada del entrenamiento.")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le = LabelEncoder()
    y = le.fit_transform(y)

    print(f"--- 2. Partición: 80% Entrenamiento / 20% Test Ciego ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    archivo_test = "datos_test_20_ciego.csv"
    X_test.to_csv(archivo_test, index=False)

    # Guardamos las soluciones para que test.py pueda dibujar la Matriz de Confusión si la pides
    archivo_soluciones = "datos_test_20_soluciones.csv"
    pd.DataFrame({'Etiqueta_Real': y_test}).to_csv(archivo_soluciones, index=False)

    print("--- 3. Construyendo Preprocesador Dinámico (Numérico + Texto) ---")
    with open(archivo_config, 'r') as f:
        config = json.load(f)["preprocessing"]

    # Detectamos automáticamente qué es número y qué es texto
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    # Tubería Numérica
    numeric_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy=config.get('impute_strategy', 'median'))),
        ('scaler', StandardScaler() if config.get('scaling') == 'standard' else 'passthrough')
    ])

    # Tubería Categórica (Texto) -> Vital para Santander
    categorical_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

    print("--- 4. Ensamblando Pipelines con Balanceo de Clases ---")

    tipo_muestreo = config.get('sampling', 'none')
    sampler_step = None
    if tipo_muestreo == 'undersampling':
        sampler_step = ('sampler', RandomUnderSampler(random_state=42))
        print("[INFO] Muestreo: Undersampling activado.")
    elif tipo_muestreo == 'oversampling':
        sampler_step = ('sampler', SMOTE(random_state=42))
        print("[INFO] Muestreo: Oversampling (SMOTE) activado.")

    # Definimos la base de las tuberías
    pasos_knn = [('preprocessor', preprocessor)]
    pasos_tree = [('preprocessor', preprocessor)]
    pasos_nb = [('preprocessor', preprocessor)]
    pasos_rf = [('preprocessor', preprocessor)]

    # Inyectamos el balanceo si el JSON lo pide
    if sampler_step:
        pasos_knn.append(sampler_step)
        pasos_tree.append(sampler_step)
        pasos_nb.append(sampler_step)
        pasos_rf.append(sampler_step)

    pasos_knn.append(('classifier', KNeighborsClassifier()))
    pasos_tree.append(('classifier', DecisionTreeClassifier(random_state=42)))
    pasos_nb.append(('classifier', GaussianNB()))
    pasos_rf.append(('classifier', RandomForestClassifier(random_state=42, n_jobs=-1)))

    pipe_knn = ImbPipeline(steps=pasos_knn)
    pipe_tree = ImbPipeline(steps=pasos_tree)
    pipe_nb = ImbPipeline(steps=pasos_nb)
    pipe_rf = ImbPipeline(steps=pasos_rf)

    print("--- 5. Definiendo Espacios de Búsqueda ---")
    parametros_knn = {
        'classifier__n_neighbors': [3, 5, 7, 9],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__p': [1, 2]
    }

    parametros_tree = {
        'classifier__max_depth': [3, 5, 10, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__criterion': ['gini', 'entropy']
    }

    parametros_nb = {
        'classifier__var_smoothing': [1e-9, 1e-7, 1e-5]
    }

    parametros_rf = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }

    metricas = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1_macro': 'f1_macro'
    }

    print("--- 6. Ejecutando Entrenamiento y Validación Cruzada ---")
    modelos_ganadores = []

    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics')

    if args.algo in ['knn', 'all']:
        print(">>> Entrenando kNN...")
        grid_knn = GridSearchCV(pipe_knn, parametros_knn, cv=5, scoring=metricas, refit='f1_macro', n_jobs=-1)
        grid_knn.fit(X_train, y_train)
        print(f"Ganador kNN: F-score {grid_knn.best_score_:.4f}")
        modelos_ganadores.append(("mejor_modelo_knn.sav", grid_knn.best_estimator_))
        guardar_metricas(grid_knn.cv_results_, "resultados_knn.csv")

    if args.algo in ['tree', 'all']:
        print(">>> Entrenando Árbol de Decisión...")
        grid_tree = GridSearchCV(pipe_tree, parametros_tree, cv=5, scoring=metricas, refit='f1_macro', n_jobs=-1)
        grid_tree.fit(X_train, y_train)
        print(f"Ganador Árbol: F-score {grid_tree.best_score_:.4f}")
        modelos_ganadores.append(("mejor_modelo_tree.sav", grid_tree.best_estimator_))
        guardar_metricas(grid_tree.cv_results_, "resultados_tree.csv")

    if args.algo in ['nb', 'all']:
        print(">>> Entrenando Naive Bayes...")
        grid_nb = GridSearchCV(pipe_nb, parametros_nb, cv=5, scoring=metricas, refit='f1_macro', n_jobs=-1)
        grid_nb.fit(X_train, y_train)
        print(f"Ganador Naive Bayes: F-score {grid_nb.best_score_:.4f}")
        modelos_ganadores.append(("mejor_modelo_nb.sav", grid_nb.best_estimator_))
        guardar_metricas(grid_nb.cv_results_, "resultados_nb.csv")

    if args.algo in ['rf', 'all']:
        print(">>> Entrenando Random Forest...")
        grid_rf = GridSearchCV(pipe_rf, parametros_rf, cv=5, scoring=metricas, refit='f1_macro', n_jobs=-1)
        grid_rf.fit(X_train, y_train)
        print(f"Ganador Random Forest: F-score {grid_rf.best_score_:.4f}")
        modelos_ganadores.append(("mejor_modelo_rf.sav", grid_rf.best_estimator_))
        guardar_metricas(grid_rf.cv_results_, "resultados_rf.csv")

    print("\n--- 7. Guardando Modelos Físicos ---")
    for nombre, modelo in modelos_ganadores:
        pickle.dump(modelo, open(nombre, 'wb'))

    print("\n[INFO] Entrenamiento finalizado con éxito.")
    print(f"[INFO] Puedes evaluar manualmente ejecutando: python test.py {archivo_test} <nombre_del_modelo.sav>")

if __name__ == "__main__":
    main()ar_metricas(grid_rf.cv_results_, "resultados_rf.csv")

    print("\n--- 7. Guardando Modelos Físicos ---")
    for nombre, modelo in modelos_ganadores:
        pickle.dump(modelo, open(nombre, 'wb'))

    print("\n[INFO] Entrenamiento finalizado con éxito.")
    print(f"[INFO] Puedes evaluar manualmente ejecutando: python test.py {archivo_test} <nombre_del_modelo.sav>")


if __name__ == "__main__":
    main()