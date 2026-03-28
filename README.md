# Práctica 3: Clasificación Automatizada (Sistemas de Ayuda a la Decisión)

Este repositorio contiene un sistema completo de Machine Learning (Entrenamiento e Inferencia) capaz de procesar datos crudos, realizar limpieza dinámica, balancear clases y ejecutar una búsqueda de hiperparámetros (GridSearchCV) sobre múltiples algoritmos de clasificación.

## 🛠️ Requisitos e Instalación

* **Python:** Versión 3.12.3
* **Dependencias:** No se requiere ningún archivo extra. Para instalar todas las librerías necesarias, abre la terminal en la raíz del proyecto y ejecuta este único comando:

```bash
pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

* **`train.py`**: Script principal de la fase de entrenamiento. Ingiere los datos crudos, los preprocesa (imputación, escalado, One-Hot Encoding), balancea las clases de forma segura (para no contaminar la validación) y entrena los modelos especificados buscando la mejor combinación de parámetros. Exporta el modelo ganador en formato `.sav`.
* **`test.py`**: Script de la fase de inferencia. Carga un modelo `.sav` pre-entrenado y un dataset ciego para generar las predicciones finales y exportarlas a un CSV limpio.
* **`configuration.json`**: Archivo de control. Permite modificar las estrategias de imputación, escalado y balanceo (ej. activar SMOTE o Undersampling) sin necesidad de alterar el código fuente.

## 🚀 Instrucciones de Ejecución

El flujo de trabajo se divide en dos fases obligatorias y secuenciales:

### Fase 1: Entrenamiento (`train.py`)

Para entrenar un modelo, se debe ejecutar el script indicando el dataset de entrenamiento, el archivo de configuración y, opcionalmente, el algoritmo deseado.

**Sintaxis básica:**
```bash
python train.py <archivo_datos.csv> -c <archivo_config.json> [--algo <algoritmo>]
```

**Opciones del argumento `--algo`:**
* `knn`: K-Nearest Neighbors
* `tree`: Decision Tree
* `nb`: Naive Bayes
* `rf`: Random Forest
* `all`: (Por defecto). Ejecuta todos los algoritmos simultáneamente, los evalúa mediante validación cruzada, los compara por su métrica F1-Macro y guarda a los ganadores de cada categoría.

**Ejemplos de uso práctico:**
```bash
# Entrenar únicamente Random Forest con un dataset de ejemplo
python train.py ejemplo.csv -c configuration.json --algo rf

# Entrenar todos los algoritmos disponibles compitiendo entre sí
python train.py ejemplo.csv -c configuration.json --algo all
```

### Fase 2: Clasificación de Nuevos Ítems (`test.py`)

Una vez generado el modelo físico (`.sav`) en la Fase 1, se utiliza este script para predecir sobre un nuevo conjunto de datos ciego (sin la columna objetivo).

**Sintaxis básica:**
```bash
python test.py <datos_nuevos_ciegos.csv> <modelo_guardado.sav>
```

**Ejemplo de uso práctico:**
```bash
python test.py datos_test.csv mejor_modelo_rf.sav
```
*Este proceso generará automáticamente un archivo llamado `predicciones_mejor_modelo_rf.csv` (el nombre se adapta al modelo introducido) que contendrá las soluciones emparejadas con su ID original, listo para su entrega y corrección.*

## ⚠️ Notas Arquitectónicas para la Evaluación

1. **Gestión Dinámica y Segura del `ID`:** Los scripts están diseñados para escanear el dataset. Si detectan una columna `ID`, la extirpan temporalmente antes del entrenamiento. Esto evita que los algoritmos utilicen el identificador del cliente como variable matemática. En la fase de test, el script restaura automáticamente la columna `ID` en el archivo final para garantizar que el CSV de predicciones mantenga el formato exacto requerido por los scripts de corrección.
2. **Pipelines para Variables Categóricas:** El preprocesador (`ColumnTransformer`) detecta dinámicamente qué columnas son de texto (categóricas) y cuáles numéricas. Aplica transformaciones de `OneHotEncoder` de forma segura a las de texto, evitando fallos de compilación al enfrentarse a datasets complejos como el de Santander.
3. **ImbPipeline:** Se ha utilizado la tubería de la librería `imbalanced-learn` en lugar de la estándar de `scikit-learn` para asegurar que el balanceo de clases (SMOTE/Undersampling) se aplique **exclusivamente** sobre los pliegues de entrenamiento durante la validación cruzada, evitando el *data leakage* en la validación.
`

## ⚖️ Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.