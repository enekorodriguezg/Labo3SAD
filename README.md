# Clasificación Automatizada (Sistemas de Apoyo a la Decisión)

Este repositorio contiene un sistema completo y defensivo de Machine Learning (Entrenamiento e Inferencia) capaz de procesar datos crudos, realizar limpieza dinámica, balancear clases y ejecutar una búsqueda de hiperparámetros (GridSearchCV) sobre múltiples algoritmos de clasificación.

## 🛠️ Requisitos e Instalación

* **Python:** Versión 3.12.3
* **Dependencias:** No se requiere ningún archivo extra. Para instalar todas las librerías necesarias, abre la terminal en la raíz del proyecto y ejecuta este único comando:

```bash
pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

* **`train.py`**: Script principal de la fase de entrenamiento. Ingiere los datos crudos, los preprocesa (imputación, escalado, One-Hot Encoding), balancea las clases de forma segura para evitar filtraciones y entrena los modelos especificados. Exporta el modelo ganador en formato `.sav` y genera el diccionario de clases (`label_encoder.sav`).
* **`test.py`**: Script de la fase de inferencia. Carga un modelo `.sav` pre-entrenado, el diccionario de clases y un dataset ciego para generar predicciones finales. Incluye mecanismos de defensa contra formatos de entrada incorrectos o sucios.
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
* `all`: (Por defecto). Ejecuta todos los algoritmos simultáneamente, evalúa mediante validación cruzada y guarda a los ganadores de cada categoría.

**Ejemplos de uso práctico:**
```bash
# Entrenar únicamente Random Forest con un dataset de ejemplo
python train.py ejemplo.csv -c configuration.json --algo rf

# Entrenar todos los algoritmos disponibles compitiendo entre sí
python train.py ejemplo.csv -c configuration.json --algo all
```

### Fase 2: Clasificación de Nuevos Ítems (`test.py`)

Una vez generados el modelo físico (`.sav`) y el `label_encoder.sav` en la Fase 1, se utiliza este script para predecir sobre un nuevo conjunto de datos.

**Sintaxis básica:**
```bash
python test.py <datos_nuevos_ciegos.csv> <modelo_guardado.sav>
```

**Ejemplo de uso práctico:**
```bash
python test.py datos_test.csv mejor_modelo_rf.sav
```
*Este proceso generará automáticamente un archivo llamado `predicciones_mejor_modelo_rf.csv` que contendrá las soluciones en formato de texto legible (no numérico), emparejadas con su ID original, listo para su entrega.*

## 🛡️ Arquitectura Defensiva y Notas para la Evaluación

El sistema ha sido reestructurado asumiendo que los datos de evaluación pueden presentar inconsistencias. Se han implementado las siguientes capas de seguridad:

1. **Extracción Dinámica de Identificadores:** El sistema no depende de nombres de columna rígidos. Escanea el dataset mediante expresiones regulares buscando variaciones de variables identificadoras (ej. `ID`, `id_cliente`, `passengerid`). Si las detecta, las aísla temporalmente para evitar que el algoritmo las procese como variables matemáticas, restaurándolas al final para la entrega de resultados.
2. **Persistencia Categórica (LabelEncoder):** La variable objetivo no se traduce destructivamente. El mapeo de clases (ej. de "Aprobado"/"Suspendido" a `0`/`1`) se serializa en `label_encoder.sav` durante el entrenamiento. En la inferencia, se invierte la transformación para que el CSV final devuelva el formato de negocio original exigido.
3. **Prevención de Feature Mismatch (Data Leakage):** Durante la inferencia (`test.py`), el script escanea el dataset de entrada contra la firma del modelo entrenado. Si el dataset ciego incluye accidentalmente la variable objetivo u otras columnas sobrantes, el sistema las extirpa automáticamente antes de la predicción, evitando el colapso por desajuste de dimensionalidad.
4. **ImbPipeline:** Se ha utilizado la tubería de la librería `imbalanced-learn` en lugar de la estándar de `scikit-learn` para asegurar que el balanceo de clases (SMOTE/Undersampling) se aplique **exclusivamente** sobre los pliegues de entrenamiento durante la validación cruzada, aislando el conjunto de validación y garantizando métricas reales.

## ⚖️ Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
