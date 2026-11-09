# Auditoría técnica — Predicción de enfermedad cardíaca con redes neuronales

| | |
|---|---|
| **Alcance** | Código original (`heart_disease_prediction_nn.py`, `README.md`, `requirements.txt`, `heart.csv`, `model_heart_attack.pth` en el commit `7ac2c31`), refactor en `src/heart_nn/` + `tests/` + `artifacts/` y API en `api/` |
| **Rama** | `refactor/auditoria-y-angular` |
| **Fecha** | 2026-11-09 |
| **Método** | Revisión manual del código, análisis del CSV con pandas, carga del `.pth` con `weights_only=True` y reproducción de la metodología original con 10 semillas |

Severidades: **Crítica** (invalida resultados o el uso del artefacto), **Alta** (sesgo o defecto importante), **Media** (riesgo moderado / deuda relevante), **Baja** (calidad, estilo, documentación menor).
Estado: **Resuelto** (verificado en el código actual), **Mitigado**, **Parcial**, **Aceptado** (riesgo asumido y documentado), **Pendiente**.

## Resumen ejecutivo

El script original tenía defectos metodológicos que invalidaban sus métricas (early stopping y evaluación sobre el mismo test, fuga del escalador, modelo de la última época, escalador no guardado y Focal Loss mal implementada). El refactor (`src/heart_nn/`) los corrige todos y es reproducible bit a bit. La etiqueta `output` del CSV de Kaggle estaba **invertida** respecto a UCI (DQ-03, Crítica); se corrigió entrenando con `enfermedad = 1 - output`, de modo que el modelo y la API devuelven ahora la probabilidad de enfermedad cardíaca. Queda una conclusión de fondo que no es un defecto de código: la red **no supera** a una regresión logística (R-01; ROC-AUC en test 0.922 frente a 0.922).

| Ámbito | Crítica | Alta | Media | Baja | Total | Resueltos / mitigados | Parciales | Pendientes / aceptados |
|---|---|---|---|---|---|---|---|---|
| Código original | 3 | 3 | 12 | 11 | 29 | 26 | 3 | 0 |
| Refactor (`src/heart_nn`) | 0 | 0 | 2 | 7 | 9 | 1 | 1 | 7 |
| API (`api/`) | 0 | 0 | 1 | 7 | 8 | 2 | 0 | 6 |

| ID | Severidad | Hallazgo | Estado |
|---|---|---|---|
| DQ-03 | Crítica | Etiqueta objetivo invertida respecto a UCI | Resuelto |
| ML-01 | Crítica | Early stopping y evaluación final sobre el mismo conjunto de test | Resuelto |
| ML-04 | Crítica | El escalador no se persiste: el `.pth` es inutilizable en inferencia | Resuelto |
| ML-02 | Alta | Se guarda el modelo de la última época, no el mejor | Resuelto |
| ML-03 | Alta | `StandardScaler` ajustado antes de dividir (data leakage) | Resuelto |
| ML-05 | Alta | `FocalLoss` se calcula sobre la BCE media del lote, no por muestra | Resuelto |
| API-01 | Media | Sin autenticación, límite de peticiones ni cabeceras de seguridad | Pendiente |
| CQ-01 | Media | Script monolítico con efectos al importar | Resuelto |
| DEP-01 | Media | `requirements.txt` sin versiones; `torchvision` innecesario | Resuelto |
| DOC-01 | Media | Diccionario de datos no coincide con el CSV | Resuelto |
| DQ-01 | Media | Fila duplicada | Resuelto |
| DQ-02 | Media | Códigos inválidos: `caa=4` (5 filas) y `thall=0` (2 filas) | Parcial |
| ML-06 | Media | `Sigmoid` + `BCELoss` en lugar de `BCEWithLogitsLoss` | Resuelto |
| ML-07 | Media | Sin semillas: resultados no reproducibles; división no estratificada | Resuelto |
| ML-08 | Media | Evaluación con un único split pequeño, sin validación cruzada, sin intervalos ni baseline | Resuelto |
| ML-09 | Media | Umbral fijo 0.5 vía `round()` y métricas incompletas | Resuelto |
| ML-10 | Media | Variables categóricas nominales tratadas como continuas | Resuelto |
| R-01 | Media | La red no supera al baseline de regresión logística | Resuelto |
| R-02 | Media | Métricas de `metrics.json` calculadas con umbrales distintos | Pendiente |
| SEC-01 | Media | Modelo serializado con pickle (`.pth`) sin pautas de carga segura | Parcial |
| SEC-02 | Media | Ausencia de aviso médico | Resuelto |
| API-02 | Baja | Rangos de entrada duplicados entre la API y el paquete del modelo | Mitigado |
| API-03 | Baja | Las entradas se limitan al rango observado en el dataset | Aceptado |
| API-04 | Baja | Umbral de clasificación y niveles de riesgo son independientes | Aceptado |
| API-05 | Baja | `class_distribution` difiere entre `/api/metrics` y `/api/dataset/summary` | Aceptado |
| API-06 | Baja | Los artefactos se cargan una sola vez al arrancar | Aceptado |
| API-07 | Baja | Mensajes de error 503 con rutas internas | Resuelto |
| API-08 | Baja | Aviso de obsolescencia en los tests | Pendiente |
| CQ-02 | Baja | Ruta relativa al directorio de trabajo y columna objetivo por posición | Resuelto |
| CQ-03 | Baja | Imports sin usar | Resuelto |
| CQ-04 | Baja | Logging con `print` en cada época | Resuelto |
| CQ-05 | Baja | Gráficos bloqueantes y no guardados | Resuelto |
| CQ-06 | Baja | API obsoleta de seaborn | Resuelto |
| CQ-07 | Baja | Estilo y tipado | Resuelto |
| DOC-02 | Baja | Afirmaciones de resultados sin cifras; Markdown roto | Resuelto |
| DQ-04 | Baja | Valores extremos sin revisar | Resuelto |
| ML-11 | Baja | Focal Loss sin justificación: las clases están casi equilibradas | Resuelto |
| ML-12 | Baja | Entrenamiento full-batch, hiperparámetros fijos en el código | Resuelto |
| R-03 | Baja | El umbral de Youden se estima con los modelos de CV pero se aplica al modelo final | Pendiente |
| R-04 | Baja | Conjunto de test muy pequeño | Parcial |
| R-05 | Baja | Selección de configuración y estimación de CV sobre los mismos datos | Pendiente |
| R-06 | Baja | `set_seed` activa algoritmos deterministas de forma global | Pendiente |
| R-07 | Baja | Artefactos sin versiones de librerías ni hash; preprocesador en pickle | Pendiente |
| R-08 | Baja | Codificación de `restecg` 0/1 inferida heurísticamente | Pendiente |
| R-09 | Baja | Los tests y el lint de la API no se ejecutan desde la raíz; sin CI ni LICENSE | Pendiente |
| REPO-01 | Baja | Higiene del repositorio | Parcial |

---

## 1. Hallazgos sobre el código original

### 1.1 Corrección del modelo y metodología de ML

#### ML-01 — Early stopping y evaluación final sobre el mismo conjunto de test
- **Severidad:** Crítica
- **Ubicación:** `heart_disease_prediction_nn.py:89-107` (bucle) y `:128-144` (reporte)
- **Descripción:** la pérdida de `X_test` decide cuándo parar (selección de modelo) y después se reportan las métricas sobre ese mismo `X_test`. No existe conjunto de validación.
- **Impacto:** las métricas publicadas están sesgadas al alza (fuga de información del test hacia la selección del modelo); no son una estimación honesta del rendimiento.
- **Recomendación:** partición train/validación/test (o validación interna dentro del train); early stopping sobre validación; el test se usa una sola vez al final.
- **Estado:** Resuelto — partición estratificada train/val/test (`data.py::split_data`); early stopping con val y test evaluado una sola vez (`train.py::run`).

#### ML-02 — Se guarda el modelo de la última época, no el mejor
- **Severidad:** Alta
- **Ubicación:** `:100-107`, `:147`
- **Descripción:** se registra `best_test_loss` pero nunca se guardan los pesos de esa época. Al parar, el modelo tiene `patience=20` épocas más de entrenamiento que el óptimo, y eso es lo que se evalúa y se guarda en `model_heart_attack.pth`.
- **Impacto:** el artefacto no corresponde al mejor punto de la curva; las métricas y el `.pth` corresponden a un modelo potencialmente sobreajustado. En la reproducción con 10 semillas el paro ocurre siempre exactamente 20 épocas después del mejor punto (p. ej. mejor época 80, paro en la 100).
- **Recomendación:** guardar una copia (`copy.deepcopy(model.state_dict())`) cuando mejora la pérdida de validación y restaurarla al terminar.
- **Estado:** Resuelto — `train.py::fit_network` guarda `best_state` y lo restaura con `load_state_dict(best_state)`; test `test_best_weights_are_restored`.

#### ML-03 — `StandardScaler` ajustado antes de dividir (data leakage)
- **Severidad:** Alta
- **Ubicación:** `:23-24` (antes del `train_test_split` de `:33`)
- **Descripción:** `fit_transform` se aplica a todo el dataset; la media y desviación incluyen las filas de test.
- **Impacto:** fuga de información (pequeña en magnitud con StandardScaler, pero metodológicamente incorrecta) y hábito peligroso con otros preprocesados (imputación, selección de variables).
- **Recomendación:** dividir primero; ajustar el preprocesador solo con train (idealmente dentro de un `Pipeline`/`ColumnTransformer`).
- **Estado:** Resuelto — `train.py::train_pipeline` ajusta el `ColumnTransformer` solo con train; test `test_preprocessor_fit_only_on_train`.

#### ML-04 — El escalador no se persiste: el `.pth` es inutilizable en inferencia
- **Severidad:** Crítica
- **Ubicación:** `:23-24`, `:147`
- **Descripción:** solo se guarda `state_dict`. El modelo espera entradas estandarizadas con medias/desviaciones que no se guardan en ninguna parte; tampoco el orden de las columnas, el umbral ni la versión.
- **Impacto:** cualquier predicción con `model_heart_attack.pth` sobre datos crudos es silenciosamente incorrecta. No hay forma reproducible de servir el modelo.
- **Recomendación:** persistir el preprocesador (`joblib`) y un `metadata.json` (orden de variables, umbral, configuración, semilla, versión) junto al modelo; exponer una clase de inferencia que los cargue juntos.
- **Estado:** Resuelto — `artifacts/preprocessor.joblib` + `metadata.json` + `HeartPredictor.from_artifacts`; verificado con la API real.

#### ML-05 — `FocalLoss` se calcula sobre la BCE media del lote, no por muestra
- **Severidad:** Alta
- **Ubicación:** `:63-67`
- **Descripción:** `nn.BCELoss()` (reducción `mean`) devuelve un escalar; `pt = exp(-BCE_media)` y el factor `(1-pt)^γ` es único para todo el lote. La Focal Loss (Lin et al., 2017) requiere `pt` por muestra para reducir el peso de los ejemplos fáciles. Además `alpha` es un escalar (no pondera clases) y `reduction` es inútil (`mean`/`sum` de un escalar) y no admite `'none'`.
- **Impacto:** la pérdida es en realidad una BCE reescalada por un factor global; no tiene el efecto que el README y los comentarios le atribuyen. Ejemplo medido sobre 8 predicciones aleatorias: BCE = 0.895, "focal" original = 0.313, focal correcta por muestra = 0.512.
- **Recomendación:** `bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")`, `pt = exp(-bce)`, `loss = (alpha_t * (1-pt)**γ * bce).mean()`; o, dado que las clases están casi equilibradas (ver ML-11), usar simplemente `BCEWithLogitsLoss`.
- **Estado:** Resuelto — `losses.py::FocalLoss` por muestra sobre logits; tests contra la fórmula manual. BCE es ahora la pérdida por defecto.

#### ML-06 — `Sigmoid` + `BCELoss` en lugar de `BCEWithLogitsLoss`
- **Severidad:** Media
- **Ubicación:** `:48`, `:64`
- **Descripción:** aplicar la sigmoide en el modelo y luego `BCELoss` es numéricamente menos estable (saturación, `log(0)` recortado a -100) que trabajar con logits.
- **Impacto:** gradientes pobres en predicciones saturadas; pérdida recortada.
- **Recomendación:** la red devuelve logits; `BCEWithLogitsLoss` en entrenamiento y `torch.sigmoid` solo en inferencia.
- **Estado:** Resuelto — `model.py::HeartClassifier` devuelve logits; `BCEWithLogitsLoss`.

#### ML-07 — Sin semillas: resultados no reproducibles; división no estratificada
- **Severidad:** Media
- **Ubicación:** `:33` (solo `random_state=42` en el split), `:114-116`
- **Descripción:** no se fija `torch.manual_seed` (inicialización de pesos y dropout) ni `numpy`. El split no usa `stratify=y`.
- **Impacto:** cada ejecución produce un modelo y unas métricas distintas. Reproduciendo la metodología original con 10 semillas: accuracy 0.839 ± 0.056 (rango 0.750–0.921), F1 0.860 ± 0.050, ROC-AUC 0.903 ± 0.031. Una única cifra de un único split no es representativa.
- **Recomendación:** función `set_seed()` (random, numpy, torch, `torch.use_deterministic_algorithms` opcional), `stratify=y`.
- **Estado:** Resuelto — `train.py::set_seed`, `stratify` en particiones y folds; reproducibilidad bit a bit verificada (ver sección 2).

#### ML-08 — Evaluación con un único split pequeño, sin validación cruzada, sin intervalos ni baseline
- **Severidad:** Media
- **Ubicación:** `:33`, `:131-144`
- **Descripción:** 76 muestras de test; sin CV, sin intervalos de confianza y sin comparar con un modelo de referencia (regresión logística), que en este dataset suele rendir igual o mejor que una MLP.
- **Impacto:** no se puede afirmar que la red aporte valor ni cuantificar la incertidumbre (ver la variabilidad de ML-07).
- **Recomendación:** k-fold estratificado (p. ej. 5×) para estimar media ± desviación; baseline de regresión logística con el mismo preprocesado; reportar ROC-AUC.
- **Estado:** Resuelto — CV estratificada 5 folds, baseline logístico (también con CV) y ROC-AUC en `metrics.json`.

#### ML-09 — Umbral fijo 0.5 vía `round()` y métricas incompletas
- **Severidad:** Media
- **Ubicación:** `:128`; `roc_curve`/`auc` importados pero no usados (`:9`)
- **Descripción:** no se calcula ROC-AUC ni se analiza el umbral. En un contexto de cribado médico el coste de un falso negativo es mayor que el de un falso positivo.
- **Impacto:** el punto de operación no está justificado; recall potencialmente subóptimo.
- **Recomendación:** reportar ROC-AUC y curva ROC; elegir/documentar el umbral (persistido en metadatos) según el objetivo (p. ej. recall mínimo) usando validación, no test.
- **Estado:** Resuelto — umbral de Youden sobre predicciones out-of-fold persistido en `metadata.json` (ver R-02 y R-03).

#### ML-10 — Variables categóricas nominales tratadas como continuas
- **Severidad:** Media
- **Ubicación:** `:18`, `:23-24`
- **Descripción:** `cp`, `restecg`, `slp` y `thall` son nominales (sus códigos no tienen orden numérico) y se estandarizan como si fueran continuas.
- **Impacto:** se impone un orden artificial; la red puede compensarlo, pero es peor práctica y empeora modelos lineales de referencia.
- **Recomendación:** `ColumnTransformer` con `OneHotEncoder(handle_unknown="ignore")` para nominales y `StandardScaler` para continuas.
- **Estado:** Resuelto — one-hot de `cp`, `restecg`, `slp`, `caa`, `thall` (`features.py::build_preprocessor`), elegido con `select.py`.

#### ML-11 — Focal Loss sin justificación: las clases están casi equilibradas
- **Severidad:** Baja
- **Ubicación:** `:54-55` (comentario), `:117`
- **Descripción:** el comentario justifica la Focal Loss por "desequilibrio de clases", pero la distribución es 165 (54.5 %) vs 138 (45.5 %) (`output=1` / `output=0`; equivalen a 165 sin enfermedad y 138 con enfermedad tras corregir DQ-03).
- **Impacto:** complejidad sin beneficio y confusión al lector.
- **Recomendación:** BCE estándar (opcionalmente `pos_weight`) salvo evidencia experimental en contra.
- **Estado:** Resuelto — BCE por defecto, decisión respaldada por `artifacts/model_selection.json`.

#### ML-12 — Entrenamiento full-batch, hiperparámetros fijos en el código
- **Severidad:** Baja
- **Ubicación:** `:72`, `:80-86`, `:115-117`, `:41-47`
- **Descripción:** sin mini-batches, sin `weight_decay`, `lr`, épocas, `patience`, tamaño de capas y dropout codificados; la arquitectura tiene `13` entradas fijas.
- **Impacto:** difícil experimentar o ajustar; acoplamiento al número de columnas.
- **Recomendación:** configuración en un dataclass/fichero; `in_features` derivado de los datos; `DataLoader` con mini-batches.
- **Estado:** Resuelto — `config.py::TrainConfig`, `DataLoader` con mini-lotes, `weight_decay`, `input_dim` derivado.

### 1.2 Calidad de datos (`heart.csv`)

Análisis: 303 filas × 14 columnas, todas numéricas, sin valores nulos.

#### DQ-01 — Fila duplicada
- **Severidad:** Media
- **Ubicación:** `heart.csv`, filas de datos 164 y 165 (índices pandas 163 y 164): `38,1,2,138,175,0,1,173,0,0.0,2,4,2,1`
- **Descripción:** registro idéntico duplicado. Además contiene `caa=4` (ver DQ-02).
- **Impacto:** si cae una copia en train y otra en test, fuga directa; ligera sobrerrepresentación.
- **Recomendación:** `drop_duplicates()` antes de dividir y documentarlo.
- **Estado:** Resuelto — `data.py::drop_duplicates` antes de partir; test `test_load_dataset_removes_duplicates`.

#### DQ-02 — Códigos inválidos: `caa=4` (5 filas) y `thall=0` (2 filas)
- **Severidad:** Media
- **Ubicación:** `heart.csv`, columnas `caa` y `thall`
- **Descripción:** en el dataset UCI Cleveland original `ca` toma 0-3 y `thal` 3/6/7 (aquí recodificado 1-3). Los valores `caa=4` y `thall=0` son artefactos conocidos de la versión de Kaggle y corresponden a valores ausentes del original.
- **Impacto:** el modelo aprende de categorías espurias; en la API habrá que decidir si se aceptan.
- **Recomendación:** tratarlos como `NaN` e imputar (moda) dentro del pipeline, o eliminarlos; documentar la decisión.
- **Estado:** Parcial — se conservan como categoría explícita "Desconocido" en el one-hot (decisión documentada en `features.py` y README); no se imputan.

#### DQ-03 — Etiqueta objetivo invertida respecto a UCI
- **Severidad:** Crítica
- **Ubicación:** `heart.csv` columna `output`; `README.md` ("0 = less chance of heart attack, 1 = more chance")
- **Descripción:** esta versión de Kaggle deriva de UCI Cleveland, donde el objetivo indica diagnóstico de enfermedad coronaria (estrechamiento > 50 %), no "probabilidad de infarto". Además, la comunidad ha documentado que en esta versión la etiqueta parece invertida respecto al original. Los datos lo apoyan: en UCI, más vasos obstruidos (`ca`) y el defecto reversible de talio se asocian a enfermedad; aquí `caa=0` tiene `output=1` en el 74 % de los casos (130/175) mientras que `caa=2` solo en el 18 % (7/38), y `thall=3` (defecto reversible) tiene `output=1` solo en el 24 % (28/117). Es decir, `output=1` se comporta como "ausencia" de enfermedad según la semántica UCI. El análisis del refactor lo confirma con más variables: los pacientes con `output=1` son más jóvenes, alcanzan mayor `thalachh`, tienen menor `oldpeak`, menos `caa` y menos `exng`, y el recuento tras deduplicar (164) coincide con el grupo "sin enfermedad" (`num = 0`) de UCI Cleveland.
- **Impacto:** las predicciones pueden interpretarse al revés o sobredimensionarse ("ataque cardíaco"). Con la etiqueta original, el modelo predecía en la práctica la probabilidad de **no** tener enfermedad, mientras la documentación decía lo contrario.
- **Recomendación:** documentar el origen (UCI Heart Disease, Cleveland) y la probable inversión; mantener la salida como P(`output=1`) pero acompañarla siempre de la advertencia; no presentar la probabilidad como "riesgo de infarto". Si se dispone del dataset UCI original, reentrenar con su etiqueta (`num > 0`).
- **Estado:** Resuelto — con la aprobación del usuario, el modelo se entrena con `enfermedad = 1 - output` (`src/heart_nn/features.py`: `LABEL = "enfermedad"`, `disease_label()`; `data.py::split_xy` devuelve `y = 1 - output`). `metadata.json` documenta `target = {name: "enfermedad", source_column: "output", definition: "enfermedad = 1 - output", positive_class: 1}`, y `/api/dataset/summary` cuenta las clases sobre la misma etiqueta. La decisión se basa en la evidencia anterior (asociaciones de `caa`, `thall`, `age`, `thalachh`, `oldpeak` y `exng` con la clase, y recuento de 164/138 tras deduplicar, idéntico a la distribución sin/con enfermedad de UCI Cleveland) y no en una fuente oficial de Kaggle; si se dispusiera del CSV de UCI debería verificarse directamente con su columna `num`.

#### DQ-04 — Valores extremos sin revisar
- **Severidad:** Baja
- **Ubicación:** `chol=564` (1 fila), `trtbps=200`, `oldpeak=6.2`, `restecg=2` (solo 4 filas)
- **Descripción:** valores plausibles pero extremos o categorías muy raras; sin validación de rangos de entrada.
- **Impacto:** escaso en este dataset; relevante para la validación de entradas de la API.
- **Recomendación:** validación de esquema/rangos al cargar (y en la API); documentar los rangos.
- **Estado:** Resuelto — `data.py::validate_schema`, `features.py::validate_record` y esquema Pydantic de la API.

### 1.3 Reproducibilidad, dependencias y seguridad

#### DEP-01 — `requirements.txt` sin versiones; `torchvision` innecesario
- **Severidad:** Media
- **Ubicación:** `requirements.txt:1-7`
- **Descripción:** ninguna dependencia fijada ni versión de Python declarada; `torchvision` (≈ cientos de MB) no se usa; `numpy` no se importa directamente (sí transitivamente).
- **Impacto:** instalaciones no reproducibles y más pesadas; posibles roturas por cambios de API (p. ej. seaborn 0.13, ver CQ-06).
- **Recomendación:** `pyproject.toml` con rangos compatibles, extras de desarrollo y, opcionalmente, un lock; eliminar `torchvision`.
- **Estado:** Resuelto — `pyproject.toml` con rangos, extras `dev`/`plots`; `torchvision` y `seaborn` eliminados.

#### SEC-01 — Modelo serializado con pickle (`.pth`) sin pautas de carga segura
- **Severidad:** Media
- **Ubicación:** `:147`, `model_heart_attack.pth`
- **Descripción:** `torch.save` usa pickle. Quien cargue un `.pth` de origen no confiable con `torch.load(..., weights_only=False)` (valor por defecto en torch < 2.6) puede ejecutar código arbitrario. Verificado: el `.pth` del repositorio es un `state_dict` benigno (6 tensores, se carga con `weights_only=True`).
- **Impacto:** riesgo de ejecución de código al reutilizar modelos de terceros.
- **Recomendación:** cargar siempre con `torch.load(path, weights_only=True, map_location="cpu")` o usar `safetensors`; publicar un hash del artefacto. Ojo: `joblib`/pickle del preprocesador tiene el mismo riesgo y no admite modo seguro (solo cargar artefactos propios).
- **Estado:** Parcial — `torch.load(..., weights_only=True)` en `predict.py::from_artifacts`; el preprocesador sigue en pickle/joblib (ver R-07).

#### SEC-02 — Ausencia de aviso médico
- **Severidad:** Media
- **Ubicación:** `README.md`
- **Descripción:** se presenta como predictor de "riesgo de enfermedades cardíacas" sin advertir que no es un dispositivo médico validado.
- **Impacto:** riesgo ético/legal si se usa para decisiones clínicas.
- **Recomendación:** aviso explícito en README y en la interfaz.
- **Estado:** Resuelto — aviso médico en README, en `api/README.md` y en la descripción OpenAPI.

#### REPO-01 — Higiene del repositorio
- **Severidad:** Baja
- **Ubicación:** raíz del repositorio
- **Descripción:** sin `.gitignore`, sin `LICENSE`, binario `.pth` versionado sin procedencia (no se sabe con qué código/semilla/escalador se generó), dataset sin atribución ni licencia, sin tests ni CI, sin linter/formatter.
- **Impacto:** riesgo de commitear entornos/cachés; reutilización legal incierta; regresiones no detectadas.
- **Recomendación:** `.gitignore`, `LICENSE`, atribución del dataset (UCI / Kaggle), artefactos generados fuera de git o con metadatos, `pytest` + `ruff` + CI (GitHub Actions).
- **Estado:** Parcial — `.gitignore`, tests (pytest), `ruff`, atribución del dataset y artefactos con metadatos; falta `LICENSE` y CI (ver R-09).

### 1.4 Calidad del código

#### CQ-01 — Script monolítico con efectos al importar
- **Severidad:** Media
- **Ubicación:** todo el fichero (sin `if __name__ == "__main__":`)
- **Descripción:** carga datos, entrena, guarda y abre ventanas de gráficos al importarse; funciones dependen de variables globales (`X_train`, `y_train`… en `train_with_early_stopping`).
- **Impacto:** imposible reutilizar o testear; no apto para servir el modelo.
- **Recomendación:** paquete con módulos (datos, modelo, entrenamiento, evaluación, inferencia) y CLI.
- **Estado:** Resuelto — paquete `src/heart_nn/` con módulos y CLIs (`python -m heart_nn.train`, etc.).

#### CQ-02 — Ruta relativa al directorio de trabajo y columna objetivo por posición
- **Severidad:** Baja
- **Ubicación:** `:14`, `:18-19`
- **Descripción:** `pd.read_csv('heart.csv')` depende del CWD; `iloc[:, :-1]` asume que la última columna es el objetivo y el orden de columnas.
- **Impacto:** fallos o resultados erróneos silenciosos si el CSV cambia.
- **Recomendación:** rutas relativas al proyecto/configurables; selección de columnas por nombre con validación de esquema.
- **Estado:** Resuelto — rutas en `config.py` relativas al proyecto y configurables por CLI; columnas por nombre con validación.

#### CQ-03 — Imports sin usar
- **Severidad:** Baja
- **Ubicación:** `:9` (`roc_curve`, `auc`)
- **Recomendación:** eliminarlos (o usarlos, ver ML-09); `ruff` lo detecta.
- **Estado:** Resuelto — `ruff check src tests` sin errores.

#### CQ-04 — Logging con `print` en cada época
- **Severidad:** Baja
- **Ubicación:** `:108-110`
- **Descripción:** hasta 1000 líneas impresas con el `repr` de tensores; la época en la que se para no se imprime.
- **Recomendación:** `logging`, frecuencia configurable, `.item()`; guardar el historial en JSON.
- **Estado:** Resuelto — `logging`; historial en `artifacts/history.json`.

#### CQ-05 — Gráficos bloqueantes y no guardados
- **Severidad:** Baja
- **Ubicación:** `:138-144`, `:150-157`, `:160-183`
- **Descripción:** `plt.show()` bloquea en ejecución no interactiva; la matriz de confusión se dibuja en una figura que solo aparece al llamar al siguiente `show`; nada se guarda en disco.
- **Recomendación:** separar el cálculo de métricas (JSON) de la visualización; guardar figuras o delegarlas al frontend.
- **Estado:** Resuelto — `plots.py` con backend `Agg` guarda PNG en `artifacts/figures/`; métricas en JSON para el frontend.

#### CQ-06 — API obsoleta de seaborn
- **Severidad:** Baja
- **Ubicación:** `:170`
- **Descripción:** `sns.barplot(..., palette=...)` sin `hue` emite `FutureWarning` en seaborn ≥ 0.13 y dejará de funcionar.
- **Recomendación:** `hue=x, legend=False` o eliminar la dependencia de seaborn.
- **Estado:** Resuelto — seaborn eliminado.

#### CQ-07 — Estilo y tipado
- **Severidad:** Baja
- **Ubicación:** `:39`, `:58` (`super(Clase, self).__init__()`), todo el fichero
- **Descripción:** sintaxis de Python 2, sin type hints ni docstrings; comentarios que contradicen el código (Focal Loss "para desequilibrio", "más capas").
- **Recomendación:** `super().__init__()`, type hints, docstrings, `ruff`.
- **Estado:** Resuelto — `super().__init__()`, type hints y docstrings.

### 1.5 Documentación (`README.md`)

#### DOC-01 — Diccionario de datos no coincide con el CSV
- **Severidad:** Media
- **Ubicación:** `README.md`, sección "About this dataset"
- **Descripción:**

  | README | CSV real | Problema |
  |---|---|---|
  | `exang` | `exng` | nombre distinto |
  | `ca` (0-3) | `caa` (0-4) | nombre y rango distintos |
  | `rest_ecg` | `restecg` | nombre distinto |
  | `thalach` | `thalachh` | nombre distinto |
  | `target` | `output` | nombre distinto |
  | `cp` valores 1-4 | `cp` valores 0-3 | codificación distinta |
  | — | `oldpeak`, `slp`, `thall` | no documentadas |
  | `chol` "fetched via BMI sensor" | colesterol sérico (análisis de sangre) | descripción errónea |
  | `sex` sin codificación | 0/1 | falta la codificación |
  | `rest_ecg` 0 = normal, 1 = ST-T, 2 = HVI | `restecg` 0 = HVI, 1 = normal, 2 = ST-T | codificación distinta |
  | — | `caa=4`, `thall=0` | códigos de dato faltante no documentados |

  Codificación real verificada por frecuencias frente a UCI Cleveland (evidencia en el docstring de `src/heart_nn/features.py`): `cp` 0 = asintomático, 1 = angina atípica, 2 = dolor no anginoso, 3 = angina típica; `restecg` 0 = hipertrofia ventricular izquierda, 1 = normal, 2 = anomalía ST-T (la asignación 0/1 es la menos segura, ver R-08); `slp` 0 = descendente, 1 = plana, 2 = ascendente; `thall` 0 = faltante, 1 = defecto fijo, 2 = normal, 3 = defecto reversible; `caa` 0-3 vasos, 4 = faltante. El README original (cp 1-4, típica = 1) era incorrecto, con lo que cualquier entrada construida a partir de él estaba mal codificada.

- **Impacto:** imposible construir entradas correctas a partir de la documentación (y por tanto usar el modelo).
- **Recomendación:** diccionario completo en español con nombre real, descripción, unidades, codificación y rango observado.
- **Estado:** Resuelto — diccionario completo en el README y en `FEATURES` (servido por `GET /api/features`); ver R-08.

#### DOC-02 — Afirmaciones de resultados sin cifras; Markdown roto
- **Severidad:** Baja
- **Ubicación:** `README.md`, secciones "Resultados", "Instalación", "Uso", índice
- **Descripción:** "alta precisión, recall y F1" sin números ni metodología (y con el sesgo de ML-01); el bloque `bash` está fuera de una valla de código; listas `-Clona` sin espacio; enlace `#About` del índice roto; mezcla de inglés y español; describe gráficos que no se guardan.
- **Recomendación:** reportar métricas reales (test + CV) generadas automáticamente; corregir el Markdown.
- **Estado:** Resuelto — README reescrito con métricas reales (test + CV + baseline) y lectura honesta.

---

## 2. Hallazgos sobre el refactor (`src/heart_nn/`, `tests/`, `artifacts/`)

Revisión realizada sobre el estado del refactor al cierre de esta auditoría (los ficheros originales se eliminaron; las referencias `:N` de la sección 1 son líneas de `heart_disease_prediction_nn.py` en el commit `7ac2c31`). No se encontraron errores funcionales en el refactor: los hallazgos siguientes son de metodología, robustez o documentación. `pytest` (30 tests) y `ruff check src tests` pasan. Verificaciones ejecutadas:

- **Reproducibilidad:** dos ejecuciones de `python -m heart_nn.train` en carpetas separadas producen `metrics.json` idénticos entre sí y con los artefactos de ese momento, y `model.pt` con el mismo SHA-256. Verificado en Windows 11, CPU, torch 2.15.0+cpu (antes de invertir la etiqueta; el cambio no afecta al determinismo).
- **Inferencia:** `HeartPredictor.from_artifacts("artifacts")` carga con `torch.load(..., weights_only=True)`; el preprocesador produce la misma matriz con enteros o flotantes (el one-hot no descarta categorías al convertir a `float`); las 303 filas del CSV se predicen sin error; entradas fuera de rango o incompletas lanzan `ValueError`.
- **Resultados (seed 42, etiqueta `enfermedad = 1 - output`):** test (n = 46, umbral 0.345) accuracy 0.826, precision 0.760, recall 0.905, F1 0.826, ROC-AUC 0.922, matriz `[[19, 6], [2, 19]]`; CV 5-fold sobre train+val (umbral 0.5) accuracy 0.836 ± 0.050, F1 0.818 ± 0.051, ROC-AUC 0.896 ± 0.028.

#### R-01 — La red no supera al baseline de regresión logística
- **Severidad:** Media
- **Ubicación:** `artifacts/metrics.json` (`cv`, `baselines`, `baselines_cv`)
- **Descripción:** CV ROC-AUC 0.896 ± 0.028 (MLP) frente a 0.902 ± 0.022 (regresión logística); accuracy CV 0.836 frente a 0.840; en test ROC-AUC 0.922 frente a 0.922 (idéntico). Las diferencias están muy por debajo de la desviación entre folds.
- **Impacto:** la complejidad de la red (y su menor interpretabilidad) no está justificada por el rendimiento. No es un defecto del código sino una conclusión que debe comunicarse.
- **Recomendación:** documentarlo explícitamente en el README y el dashboard; valorar la regresión logística (calibrada e interpretable) como modelo de referencia o de producción.
- **Estado:** Resuelto (documentado en el README, sección "Lectura honesta")

#### R-02 — Métricas de `metrics.json` calculadas con umbrales distintos
- **Severidad:** Media
- **Ubicación:** `src/heart_nn/train.py` — `cross_validate` (CV con 0.5), `run` (test con el umbral de Youden; baseline con 0.5)
- **Descripción:** `test` usa el umbral 0.345, mientras que `cv`, `baselines` y `baselines_cv` usan 0.5. Existe `test_threshold_0_5`, pero un consumidor que compare `test` con `baselines.logistic_regression` mezcla umbrales.
- **Impacto:** comparaciones engañosas (p. ej. recall 0.905 de la red frente a 0.810 de la regresión logística se debe al umbral, no al modelo: con 0.5 la red obtiene recall 0.762 y F1 0.780, por debajo del baseline).
- **Recomendación:** añadir el umbral usado a cada bloque (`"threshold": ...`) o evaluar el baseline también con su propio umbral de Youden OOF; el frontend debería comparar `test_threshold_0_5` con `baselines`.
- **Estado:** Pendiente

#### R-03 — El umbral de Youden se estima con los modelos de CV pero se aplica al modelo final
- **Severidad:** Baja
- **Ubicación:** `src/heart_nn/train.py` — `run` (`youden_threshold(y_dev, oof)` y posterior `train_pipeline` final)
- **Descripción:** las probabilidades out-of-fold proceden de 5 redes distintas (cada una con su propio early stopping); el modelo final se entrena con otra partición y otra época óptima, por lo que su calibración puede diferir.
- **Impacto:** el umbral puede no ser óptimo para el modelo servido. Tras invertir la etiqueta el umbral es 0.345, bastante lejos de 0.5, por lo que la calibración del modelo final importa más.
- **Recomendación:** documentarlo, o verificar la calibración del modelo final (curva de fiabilidad / Brier score) y, opcionalmente, calibrar (Platt) con validación.
- **Estado:** Pendiente

#### R-04 — Conjunto de test muy pequeño
- **Severidad:** Baja
- **Ubicación:** `src/heart_nn/config.py` (`test_size=0.15`) → 46 muestras
- **Descripción:** con n = 46, el intervalo de confianza al 95 % de la accuracy es aproximadamente ± 0.12.
- **Impacto:** las métricas de test por sí solas no son concluyentes; las de CV son más fiables.
- **Recomendación:** presentar la CV como métrica principal y el test como comprobación; añadir IC bootstrap a `metrics.json`.
- **Estado:** Parcial — el README advierte que la CV es más fiable; faltan IC bootstrap

#### R-05 — Selección de configuración y estimación de CV sobre los mismos datos
- **Severidad:** Baja
- **Ubicación:** `src/heart_nn/select.py` (`main`), `src/heart_nn/config.py` (docstring de `TrainConfig`)
- **Descripción:** la configuración por defecto (BCE + one-hot) se eligió con CV sobre train+val y después se reporta la CV de esa misma configuración sobre los mismos datos.
- **Impacto:** sesgo optimista leve (solo 4 configuraciones comparadas). El test, que no se usó en la selección, sigue siendo una estimación honesta.
- **Recomendación:** mencionarlo en el README o usar CV anidada si se amplía la búsqueda de hiperparámetros.
- **Estado:** Pendiente

#### R-06 — `set_seed` activa algoritmos deterministas de forma global
- **Severidad:** Baja
- **Ubicación:** `src/heart_nn/train.py` — `set_seed`
- **Descripción:** `torch.use_deterministic_algorithms(True)` cambia el estado global del proceso. Usado como librería (o en GPU, donde exige `CUBLAS_WORKSPACE_CONFIG`) puede provocar errores en operaciones sin implementación determinista.
- **Recomendación:** hacerlo opcional (`deterministic: bool` en `TrainConfig`) o restaurar el estado previo al terminar; documentar la variable de entorno para GPU.
- **Estado:** Pendiente

#### R-07 — Artefactos sin versiones de librerías ni hash; preprocesador en pickle
- **Severidad:** Baja
- **Ubicación:** `src/heart_nn/train.py` — `run` (construcción de `metadata`); `src/heart_nn/predict.py` — `HeartPredictor.from_artifacts` (`joblib.load`)
- **Descripción:** `metadata.json` no registra las versiones de `torch`/`scikit-learn`/`numpy` ni un hash de `model.pt`/`preprocessor.joblib`. `joblib.load` deserializa pickle (no hay modo seguro) y un `ColumnTransformer` serializado con otra versión de scikit-learn puede fallar o comportarse distinto (`InconsistentVersionWarning`).
- **Impacto:** artefactos difíciles de auditar entre entornos; riesgo de ejecución de código si se cargan artefactos de terceros.
- **Recomendación:** añadir `library_versions` y `sha256` a los metadatos y verificarlos en `from_artifacts`; alternativamente, persistir medias/desviaciones/categorías en JSON en lugar de pickle.
- **Estado:** Pendiente

#### R-08 — Codificación de `restecg` 0/1 inferida heurísticamente
- **Severidad:** Baja
- **Ubicación:** `src/heart_nn/features.py` (docstring del módulo y opciones de `restecg`)
- **Descripción:** la asignación 0 = hipertrofia ventricular izquierda / 1 = normal se dedujo por la proporción de clases frente a UCI, que a su vez depende de la dirección de la etiqueta (DQ-03). Las demás recodificaciones (`cp`, `slp`, `thall`, `caa`) están bien respaldadas por las frecuencias.
- **Impacto:** las etiquetas mostradas al usuario para `restecg` podrían estar intercambiadas.
- **Recomendación:** indicar la incertidumbre en `description_es` / la UI.
- **Estado:** Pendiente

#### R-09 — Los tests y el lint de la API no se ejecutan desde la raíz; sin CI ni LICENSE
- **Severidad:** Baja
- **Ubicación:** `pyproject.toml` (`[tool.pytest.ini_options] testpaths = ["tests"]`, `[tool.ruff] extend-exclude = ["api", ...]`); raíz del repositorio
- **Descripción:** `pytest` en la raíz solo ejecuta `tests/` (30 tests); los 30 de `api/tests` requieren `pytest api/tests`. No hay workflow de CI que ejecute ambos, y `pyproject.toml` declara licencia MIT pero no existe el fichero `LICENSE`.
- **Recomendación:** añadir `api/tests` a `testpaths` (o un job de CI por componente) y crear `LICENSE`.
- **Estado:** Pendiente

## 3. Hallazgos sobre la API (`api/`)

Verificación de extremo a extremo con `uvicorn api.main:app --port 8000` y los artefactos reales: los seis endpoints responden según el contrato; `POST /api/predict` devuelve 422 para valores fuera de rango, tipos incorrectos (modo estricto: rechaza `"1"`, `true` y `50.5` en campos enteros) y campos adicionales; el preflight CORS desde `http://localhost:4200` se acepta y los orígenes no configurados no reciben cabecera CORS. Arrancada con `ARTIFACTS_DIR` inexistente, `/api/health` responde `model_loaded: false` y los endpoints dependientes de artefactos devuelven 503.

#### API-01 — Sin autenticación, límite de peticiones ni cabeceras de seguridad
- **Severidad:** Media
- **Ubicación:** `api/main.py`
- **Descripción:** la API está pensada para uso local junto al frontend. `/api/predict` no tiene autenticación, rate limiting ni TLS propio.
- **Impacto:** si se expone públicamente, abuso de cómputo y uso no supervisado de un modelo no validado clínicamente.
- **Recomendación:** desplegar tras un proxy inverso (TLS, rate limiting, cabeceras de seguridad); API key si se publica.
- **Estado:** Pendiente (aceptado para uso local)

#### API-02 — Rangos de entrada duplicados entre la API y el paquete del modelo
- **Severidad:** Baja
- **Ubicación:** `api/schemas.py` (`PatientFeatures`) y `src/heart_nn/features.py` (`FEATURES`)
- **Descripción:** el esquema Pydantic se declara estáticamente (para que el OpenAPI sea explícito y la API importe sin el paquete) y repite los rangos de `FEATURES`.
- **Impacto:** riesgo de desincronización.
- **Recomendación / mitigación:** `api/tests/test_api.py::test_schema_ranges_match_model_feature_catalog` falla si difieren; además `HeartPredictor.predict` revalida (`ValueError` → 422).
- **Estado:** Mitigado

#### API-03 — Las entradas se limitan al rango observado en el dataset
- **Severidad:** Baja
- **Ubicación:** `api/schemas.py`
- **Descripción:** p. ej. `age` 29-77, `chol` 126-564. Un paciente de 80 años recibe 422.
- **Impacto:** decisión deliberada (evitar extrapolar fuera del dominio de entrenamiento), pero puede sorprender al usuario.
- **Recomendación:** el frontend debe mostrar los rangos de `GET /api/features` y explicar el motivo.
- **Estado:** Aceptado

#### API-04 — Umbral de clasificación y niveles de riesgo son independientes
- **Severidad:** Baja
- **Ubicación:** `api/schemas.py` (`risk_level_for`), `artifacts/metadata.json` (`threshold`)
- **Descripción:** `prediction` usa el umbral del modelo (0.345) y `risk_level` cortes fijos 0.33/0.66 definidos por contrato; p. ej. p = 0.34 → `prediction=0`, `risk_level="moderado"`, y p = 0.50 → `prediction=1`, `risk_level="moderado"`. Las probabilidades de la red no están calibradas (R-03), por lo que los niveles son orientativos.
- **Recomendación:** mostrar ambos valores con su explicación en la UI.
- **Estado:** Aceptado (documentado)

#### API-05 — `class_distribution` difiere entre `/api/metrics` y `/api/dataset/summary`
- **Severidad:** Baja
- **Ubicación:** `api/services.py` (`compute_dataset_summary`), `src/heart_nn/train.py` (`run`, `class_distribution`)
- **Descripción:** ambos usan ya la etiqueta `enfermedad = 1 - output`, pero el resumen del dataset se calcula sobre el CSV crudo (303 filas, `{"0":165,"1":138}`, `n_duplicates: 1`) y `metrics.json` tras deduplicar (302 filas, `{"0":164,"1":138}`); la fila duplicada es de clase 0 (sin enfermedad).
- **Impacto:** cifras aparentemente contradictorias en el dashboard.
- **Recomendación:** el frontend debe etiquetar "dataset original" frente a "dataset de entrenamiento (sin duplicados)".
- **Estado:** Aceptado (documentado)

#### API-06 — Los artefactos se cargan una sola vez al arrancar
- **Severidad:** Baja
- **Ubicación:** `api/main.py` (lifespan)
- **Descripción:** tras reentrenar es necesario reiniciar la API para servir el nuevo modelo.
- **Recomendación:** reinicio en el despliegue o endpoint de recarga protegido.
- **Estado:** Aceptado

#### API-07 — Mensajes de error 503 con rutas internas
- **Severidad:** Baja
- **Ubicación:** `api/services.py` (`load_state`)
- **Descripción:** la primera versión de la API devolvía en `detail` el texto de la excepción, incluidas rutas absolutas del servidor.
- **Impacto:** divulgación de información.
- **Recomendación:** mensajes genéricos al cliente y detalle solo en el log.
- **Estado:** Resuelto (corregido durante la auditoría; cubierto por `test_degraded_errors_do_not_leak_paths`)

#### API-08 — Aviso de obsolescencia en los tests
- **Severidad:** Baja
- **Ubicación:** `api/tests` (`fastapi.testclient`)
- **Descripción:** Starlette 1.10 emite `StarletteDeprecationWarning` por usar `httpx` con `TestClient`.
- **Recomendación:** migrar el cliente de tests cuando la alternativa sea estable.
- **Estado:** Pendiente

---

## 4. Anexo: reproducción de la metodología original

Script equivalente al original (mismo modelo, misma "focal loss", early stopping sobre test con `patience=20`) ejecutado con semillas 0-9 (semilla del split = semilla de torch):

| Métrica | Media | Desv. | Mín. | Máx. |
|---|---|---|---|---|
| Época de parada | 109.3 | 40.1 | 75 | 216 |
| Mejor época (no guardada) | 89.3 | 40.1 | 55 | 196 |
| Accuracy (test) | 0.839 | 0.056 | 0.750 | 0.921 |
| F1 (test) | 0.860 | 0.050 | 0.782 | 0.923 |
| ROC-AUC (test) | 0.903 | 0.031 | 0.847 | 0.942 |

Conclusión: el rendimiento depende fuertemente de la partición y la inicialización; una cifra única obtenida con el script original no es representativa, y el modelo guardado corresponde siempre a 20 épocas después de su mejor punto.

