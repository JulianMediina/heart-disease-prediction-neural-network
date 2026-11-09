/** Texto por defecto si /api/health no trae `target_note`. */
export const DEFAULT_TARGET_NOTE =
  'La predicción es la probabilidad de enfermedad cardíaca estimada por un modelo educativo; no sustituye un diagnóstico médico.';

/** Nota breve sobre la etiqueta objetivo del dataset. */
export const LABEL_NOTE =
  'La columna output del CSV de Kaggle está invertida respecto a UCI; el modelo se entrena con enfermedad = 1 − output.';
