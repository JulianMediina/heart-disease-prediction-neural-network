# Predicción de enfermedad cardíaca con redes neuronales

Clasificador binario (perceptrón multicapa en PyTorch) entrenado sobre el dataset
*Heart Disease* (Cleveland, 303 pacientes) con un pipeline reproducible: validación
de esquema, deduplicación, partición estratificada train/val/test, preprocesamiento
ajustado solo con entrenamiento, early stopping sobre validación, validación cruzada,
baseline de regresión logística y artefactos listos para inferencia. Incluye una API
REST (FastAPI) y una interfaz web (Angular).

> **Aviso médico.** Este proyecto es exclusivamente educativo. El modelo se entrenó con
> 302 pacientes de un único hospital de los años 80 y **no** es un dispositivo médico:
> no debe usarse para diagnosticar, descartar ni tratar ninguna enfermedad. Ante
> cualquier síntoma consulte a un profesional sanitario.

## Estructura

```
├── data/heart.csv            # Dataset original (303 filas, 1 duplicada)
├── src/heart_nn/             # Paquete de ML
│   ├── config.py             # Hiperparámetros (dataclasses) y rutas
│   ├── data.py               # Carga, validación de esquema, deduplicación, partición
│   ├── features.py           # Diccionario de variables (FEATURES) y preprocesador
│   ├── model.py              # MLP 13(→27 con one-hot) → 64 → 32 → 1 (logits)
│   ├── losses.py             # BCE con logits y Focal Loss por muestra
│   ├── train.py              # CLI de entrenamiento: python -m heart_nn.train
│   ├── select.py             # Comparación de configuraciones: python -m heart_nn.select
│   ├── evaluate.py           # Métricas; CLI de re-evaluación: python -m heart_nn.evaluate
│   ├── predict.py            # HeartPredictor (inferencia sobre registros crudos)
│   └── plots.py              # Gráficos: python -m heart_nn.plots
├── artifacts/                # Artefactos versionados que consume la API
│   ├── model.pt              # state_dict de la red
│   ├── preprocessor.joblib   # ColumnTransformer ajustado (escalado + one-hot)
│   ├── metadata.json         # variables, umbral, configuración, semilla, versión
│   ├── metrics.json          # test, CV, baseline, matriz de confusión, curva ROC
│   ├── history.json          # pérdidas por época y mejor época
│   └── model_selection.json  # resultados de python -m heart_nn.select
├── tests/                    # Tests del paquete de ML (pytest)
├── api/                      # Backend FastAPI
├── web/                      # Frontend Angular
└── docs/AUDIT.md             # Auditoría del proyecto original
```

## Instalación

Requiere Python ≥ 3.10 (probado con 3.13).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # torch solo CPU
pip install -e ".[dev]"                                              # paquete + pytest, ruff, matplotlib
```

## Entrenamiento

```bash
python -m heart_nn.train                 # configuración por defecto (BCE + one-hot, semilla 42)
python -m heart_nn.train --loss focal    # Focal Loss
python -m heart_nn.train --no-one-hot    # codificación ordinal de las categóricas
python -m heart_nn.train --help
```

El comando:

1. Lee `data/heart.csv`, valida columnas/rangos/categorías y elimina la fila duplicada
   (antes de partir, para que no acabe a la vez en train y test).
2. Divide de forma estratificada en train (70 %), validación (15 %) y test (15 %).
3. Hace validación cruzada estratificada de 5 folds sobre train+val (cada fold reserva su
   propia validación interna para early stopping) y fija el umbral de decisión con el
   índice de Youden sobre las predicciones *out-of-fold*.
4. Entrena el modelo final: el preprocesador se ajusta **solo** con train, early stopping
   con la pérdida de validación (paciencia 30) y se restauran los pesos de la mejor época.
5. Evalúa **una única vez** sobre test, entrena el baseline de regresión logística y
   escribe los artefactos en `artifacts/`.

Todo es determinista con la semilla (`random`, `numpy`, `torch`, `DataLoader`, particiones
y folds): dos ejecuciones producen exactamente los mismos pesos y métricas.

Otros comandos:

```bash
python -m heart_nn.select     # compara bce/focal × one-hot/ordinal (CV 5 folds × 5 semillas)
python -m heart_nn.evaluate   # re-evalúa los artefactos sobre la partición de test
python -m heart_nn.plots      # genera PNG en artifacts/figures/
pytest                        # tests
ruff check src tests          # lint
```

### Inferencia desde Python

```python
from heart_nn.predict import HeartPredictor

p = HeartPredictor.from_artifacts("artifacts")
p.predict({"age": 63, "sex": 1, "cp": 3, "trtbps": 145, "chol": 233, "fbs": 1, "restecg": 0,
           "thalachh": 150, "exng": 0, "oldpeak": 2.3, "slp": 0, "caa": 0, "thall": 1})
# {'probability': 0.61..., 'prediction': 1}
```

`predict` / `predict_batch` lanzan `ValueError` si falta o sobra alguna variable, si un
valor no es numérico o si está fuera del rango / categorías del diccionario.

## API y web

```bash
# Backend (desde la raíz del repo, con el entorno activado)
pip install -r api/requirements.txt
uvicorn api.main:app --reload            # http://localhost:8000  (docs en /docs)

# Frontend
cd web && npm install && npm start       # http://localhost:4200 (proxy /api → :8000)
```

Consulte `api/README.md` y `web/README.md` para más detalles.

## Resultados

Partición: 210 train / 46 validación / 46 test (302 pacientes tras deduplicar;
clases 164 sin enfermedad / 138 con enfermedad). Semilla 42. Umbral 0,345 (Youden, CV).
La clase positiva es **enfermedad = 1 − `output`** (ver nota sobre la variable objetivo).

| Modelo | Conjunto | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| MLP (BCE, one-hot) | Test (n=46) | 0,826 | 0,760 | 0,905 | 0,826 | 0,922 |
| MLP (BCE, one-hot) | CV 5 folds (media ± std) | 0,836 ± 0,050 | 0,846 ± 0,093 | 0,795 ± 0,036 | 0,818 ± 0,051 | 0,896 ± 0,028 |
| Regresión logística | Test (n=46) | 0,804 | 0,773 | 0,810 | 0,791 | 0,922 |
| Regresión logística | CV 5 folds (media ± std) | 0,840 ± 0,034 | 0,849 ± 0,060 | 0,795 ± 0,038 | 0,820 ± 0,036 | 0,902 ± 0,022 |

Matriz de confusión en test (umbral 0,345): `[[TN=19, FP=6], [FN=2, TP=19]]`.
Con umbral 0,5 el test da accuracy 0,804 y F1 0,780.

**Lectura honesta:** la red no supera a una regresión logística; las diferencias están
muy por debajo de la desviación entre folds. Con 302 muestras y 13 variables, un modelo
lineal es igual de bueno y más interpretable. El test (46 pacientes) es pequeño, así que
la estimación de la CV es más fiable que la cifra puntual de test.

### Decisiones de diseño (con evidencia)

Comparación con `python -m heart_nn.select` (CV estratificada 5 folds × 5 semillas = 25
ajustes por configuración, solo sobre train+val):

| Pérdida | Categóricas | ROC-AUC | F1 | Accuracy |
|---|---|---|---|---|
| **BCE** | **one-hot** | **0,921 ± 0,045** | **0,857 ± 0,049** | **0,844 ± 0,054** |
| BCE | ordinal | 0,907 ± 0,047 | 0,845 ± 0,057 | 0,825 ± 0,066 |
| Focal (γ=2) | one-hot | 0,917 ± 0,047 | 0,854 ± 0,057 | 0,840 ± 0,063 |
| Focal (γ=2) | ordinal | 0,904 ± 0,048 | 0,847 ± 0,059 | 0,826 ± 0,068 |

* **Pérdida BCE por defecto.** Las clases están casi equilibradas (138/164), que es
  justo el caso en que Focal Loss no aporta nada; además BCE obtiene métricas
  ligeramente mejores y produce probabilidades mejor calibradas. Focal sigue
  disponible con `--loss focal` (ahora implementada correctamente, por muestra).
* **One-hot para `cp`, `restecg`, `slp`, `caa`, `thall`.** Son nominales (sus códigos
  no tienen un orden clínico coherente, ver diccionario) y la codificación one-hot
  mejora el ROC-AUC en ~0,014 tanto con BCE como con Focal. Las binarias (`sex`, `fbs`,
  `exng`) y las numéricas se estandarizan.
* **Umbral** elegido con Youden sobre predicciones out-of-fold (nunca con test) y
  guardado en `metadata.json`.

### Correcciones respecto a la versión original

| Problema original | Solución |
|---|---|
| Focal Loss aplicaba el factor focal a la BCE *media* del lote | Focal Loss por muestra sobre logits (`binary_cross_entropy_with_logits(reduction="none")`); test contra la fórmula manual |
| `StandardScaler` ajustado con todo el dataset antes de partir (fuga) | `ColumnTransformer` ajustado solo con train |
| Early stopping con el conjunto de test | Partición train/val/test estratificada; early stopping con val; test usado una vez |
| Se guardaba el modelo de la última época | Se restauran los pesos de la mejor época de validación |
| Escalador no guardado (modelo inutilizable en inferencia) | `preprocessor.joblib` + `metadata.json` + `HeartPredictor` |
| Sin semillas, partición no estratificada | Semillas globales, algoritmos deterministas, partición y folds estratificados |
| Fila duplicada en el CSV | Deduplicación antes de partir |
| Sigmoid + `BCELoss` | Logits + `BCEWithLogitsLoss` (numéricamente estable) |
| Imports sin usar, `torchvision`, gráficos con `plt.show()` dentro del entrenamiento | Dependencias mínimas; gráficos en `heart_nn.plots` |
| Sin CV, sin baseline, sin ROC-AUC | CV estratificada (media ± std), baseline logístico, ROC-AUC y curva ROC |

## Diccionario de datos

Columnas de `data/heart.csv` (versión de Kaggle del dataset Cleveland de UCI). Los códigos
de las categóricas **no** coinciden con los de UCI y el diccionario que suele acompañar al
CSV (y el de la versión anterior de este README) es incorrecto. Las equivalencias de abajo
se obtuvieron comparando las frecuencias de cada código con las del dataset UCI (detalle
en `src/heart_nn/features.py`).

| Columna | Descripción | Tipo | Valores / rango en el CSV |
|---|---|---|---|
| `age` | Edad (años) | numérica | 29 – 77 |
| `sex` | Sexo | binaria | 0 = mujer, 1 = hombre |
| `cp` | Tipo de dolor torácico | categórica | 0 = asintomático, 1 = angina atípica, 2 = dolor no anginoso, 3 = angina típica |
| `trtbps` | Presión arterial sistólica en reposo (mm Hg) | numérica | 94 – 200 |
| `chol` | Colesterol sérico (mg/dl) | numérica | 126 – 564 |
| `fbs` | Glucemia en ayunas > 120 mg/dl | binaria | 0 = no, 1 = sí |
| `restecg` | ECG en reposo | categórica | 0 = hipertrofia ventricular izquierda, 1 = normal, 2 = anomalía ST-T |
| `thalachh` | Frecuencia cardíaca máxima alcanzada (lpm) | numérica | 71 – 202 |
| `exng` | Angina inducida por ejercicio | binaria | 0 = no, 1 = sí |
| `oldpeak` | Depresión del ST inducida por ejercicio respecto al reposo (mm) | numérica | 0,0 – 6,2 |
| `slp` | Pendiente del ST en el pico de esfuerzo | categórica | 0 = descendente, 1 = plana, 2 = ascendente |
| `caa` | Nº de vasos principales coloreados por fluoroscopia | categórica | 0 – 3; 4 = dato faltante en el origen |
| `thall` | Gammagrafía de perfusión con talio | categórica | 0 = dato faltante, 1 = defecto fijo, 2 = normal, 3 = defecto reversible |
| `output` | Variable objetivo del CSV | binaria | 0 = con enfermedad, 1 = sin enfermedad (ver nota) |

**Nota sobre la variable objetivo.** El CSV de Kaggle describe `output = 1` como "mayor
probabilidad de ataque cardíaco", pero los datos muestran que la etiqueta está **invertida**
respecto a UCI: los pacientes con `output = 1` son más jóvenes, alcanzan mayor frecuencia
cardíaca máxima, tienen menos depresión del ST, menos vasos afectados y menos angina de
esfuerzo; y el recuento (164 tras deduplicar) coincide con el de pacientes **sin**
enfermedad (`num = 0`) en UCI. Por eso el proyecto entrena con
**`enfermedad = 1 − output`** (`heart_nn.features.LABEL`), y el modelo devuelve la
probabilidad de enfermedad cardíaca. Ver `docs/AUDIT.md`, hallazgo DQ-03.

Fuente: Detrano, R. et al. (1989), *UCI Heart Disease Data Set — Cleveland*;
versión redistribuida en Kaggle como `heart.csv`.
