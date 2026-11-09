/** Tipos del contrato de la API (FastAPI en `api/`). */

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_version: string | null;
  /** Advertencia sobre la semántica de la etiqueta (puede no venir). */
  target_note?: string | null;
}

export interface FeatureOption {
  value: number;
  label_es: string;
}

export type FeatureType = 'numeric' | 'categorical';

export interface FeatureSpec {
  name: FeatureName;
  label_es: string;
  description_es: string;
  type: FeatureType;
  min: number;
  max: number;
  step: number;
  options?: FeatureOption[];
}

export interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
}

export type MetricKey = keyof ClassificationMetrics;

export interface MeanStd {
  mean: number;
  std: number;
}

export interface MetricsResponse {
  test: ClassificationMetrics;
  /** Métricas de test con el umbral por defecto 0,5 (opcional). */
  test_threshold_0_5?: ClassificationMetrics;
  cv: {
    accuracy: MeanStd;
    f1: MeanStd;
    roc_auc: MeanStd;
    precision?: MeanStd;
    recall?: MeanStd;
  };
  cv_folds?: number;
  /** [[tn, fp], [fn, tp]] */
  confusion_matrix: [[number, number], [number, number]];
  roc_curve: { fpr: number[]; tpr: number[] };
  class_distribution: Record<string, number>;
  baselines: { logistic_regression: ClassificationMetrics };
  baselines_cv?: { logistic_regression: Partial<Record<MetricKey, MeanStd>> };
  n_samples: { train: number; val: number; test: number };
  /** Umbral de decisión elegido (Youden en validación cruzada). */
  threshold?: number;
}

export interface TrainingHistory {
  train_loss: number[];
  val_loss: number[];
  /** Índice (base 0) de la época con menor pérdida de validación. */
  best_epoch: number;
}

export interface FeatureStats {
  mean: number;
  std: number;
  min: number;
  max: number;
}

export interface DatasetSummary {
  n_samples: number;
  n_duplicates: number;
  class_distribution: Record<string, number>;
  features: Record<string, FeatureStats>;
}

export const FEATURE_NAMES = [
  'age',
  'sex',
  'cp',
  'trtbps',
  'chol',
  'fbs',
  'restecg',
  'thalachh',
  'exng',
  'oldpeak',
  'slp',
  'caa',
  'thall',
] as const;

export type FeatureName = (typeof FEATURE_NAMES)[number];

export type PatientInput = Record<FeatureName, number>;

export type RiskLevel = 'bajo' | 'moderado' | 'alto';

export interface PredictionResponse {
  probability: number;
  prediction: 0 | 1;
  risk_level: RiskLevel;
}

/** Detalle de error de validación de FastAPI (422). */
export interface ValidationErrorDetail {
  loc: (string | number)[];
  msg: string;
  type: string;
}
