# API REST — Predicción de enfermedad cardíaca

Backend FastAPI que expone el modelo entrenado por el paquete `heart_nn` (`src/heart_nn/`).

> Uso exclusivamente educativo. No es un dispositivo médico ni sustituye el criterio clínico.

## Instalación (Windows / PowerShell)

```powershell
python -m venv api\.venv
api\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .                    # paquete del modelo, desde la raíz del repo
pip install -r api\requirements.txt
```

La API necesita los artefactos del entrenamiento en `artifacts/` (`model.pt`,
`preprocessor.joblib`, `metadata.json`, `metrics.json`, `history.json`) y el dataset en
`data/heart.csv`. Si faltan, arranca igualmente en modo degradado: `/api/health` responde
`model_loaded: false` y el resto de endpoints devuelven **503**.

## Ejecución

```powershell
uvicorn api.main:app --port 8000          # desde la raíz del repositorio
```

Documentación interactiva: <http://localhost:8000/docs>.

### Configuración (variables de entorno o `.env`)

| Variable        | Por defecto              | Descripción                              |
|-----------------|--------------------------|------------------------------------------|
| `ARTIFACTS_DIR` | `<repo>/artifacts`       | Carpeta de artefactos del modelo         |
| `DATA_PATH`     | `<repo>/data/heart.csv`  | CSV usado por `/api/dataset/summary`     |
| `CORS_ORIGINS`  | `http://localhost:4200`  | Orígenes permitidos, separados por comas |

## Endpoints

| Método | Ruta                       | Descripción |
|--------|----------------------------|-------------|
| GET    | `/api/health`              | `{status, model_loaded, model_version, target_note}` |
| GET    | `/api/features`            | Catálogo de las 13 variables (etiquetas, rangos, opciones) |
| GET    | `/api/metrics`             | Contenido de `metrics.json` |
| GET    | `/api/training-history`    | Contenido de `history.json` (curvas de pérdida) |
| GET    | `/api/dataset/summary`     | Nº de muestras, duplicados, distribución de clases y estadísticos por variable |
| POST   | `/api/predict`             | Predicción para un paciente |

Ejemplo de predicción:

```bash
curl -X POST http://localhost:8000/api/predict -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trtbps":145,"chol":233,"fbs":1,"restecg":0,"thalachh":150,"exng":0,"oldpeak":2.3,"slp":0,"caa":0,"thall":1}'
# {"probability":0.615...,"prediction":1,"risk_level":"moderado"}
```

`risk_level`: `bajo` (< 0.33), `moderado` (< 0.66), `alto` (≥ 0.66). Las entradas con
campos ausentes, adicionales, de tipo incorrecto o fuera de rango devuelven **422**.

## Variable objetivo

El modelo devuelve la **probabilidad de enfermedad cardíaca**. La columna `output` del CSV
de Kaggle está invertida respecto al dataset UCI Cleveland original (los pacientes con
`output = 1` son más jóvenes, alcanzan mayor frecuencia cardíaca máxima y tienen menos
vasos afectados), por lo que el entrenamiento usa `enfermedad = 1 - output`.
`prediction = 1` significa "enfermedad" y `/api/dataset/summary` cuenta las clases con
la misma etiqueta. `GET /api/health` incluye la nota de `metadata.json` en `target_note`.
Detalles en `docs/AUDIT.md` (hallazgo DQ-03).

## Tests

```powershell
python -m pytest api/tests -q
```

Los tests usan un predictor simulado y artefactos falsos, por lo que no requieren entrenar
el modelo.
