import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { RouterLink } from '@angular/router';
import { forkJoin } from 'rxjs';

import { MetricKey, MetricsResponse, TrainingHistory } from '../../core/models/api.models';
import { ApiService } from '../../core/services/api.service';
import { loadable } from '../../core/services/loadable';
import { ClassDistribution } from '../../shared/components/class-distribution';
import { ComparisonBars, ComparisonRow } from '../../shared/components/comparison-bars';
import { ConfusionMatrix } from '../../shared/components/confusion-matrix';
import { ErrorBanner } from '../../shared/components/error-banner';
import { KpiCard } from '../../shared/components/kpi-card';
import { fmtNum, LineChart, LineSeries } from '../../shared/components/line-chart';
import { Skeleton } from '../../shared/components/skeleton';

export const METRIC_LABELS: Record<MetricKey, string> = {
  accuracy: 'Exactitud',
  precision: 'Precisión',
  recall: 'Sensibilidad (recall)',
  f1: 'F1',
  roc_auc: 'ROC-AUC',
};

const METRIC_ORDER: MetricKey[] = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'];

interface DashboardData {
  metrics: MetricsResponse;
  history: TrainingHistory;
}

@Component({
  selector: 'app-dashboard',
  imports: [
    DecimalPipe,
    RouterLink,
    KpiCard,
    LineChart,
    ConfusionMatrix,
    ClassDistribution,
    ComparisonBars,
    ErrorBanner,
    Skeleton,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './dashboard.html',
})
export class Dashboard {
  private readonly api = inject(ApiService);
  protected readonly data = loadable<DashboardData>(() =>
    forkJoin({ metrics: this.api.metrics(), history: this.api.trainingHistory() }),
  );

  protected readonly ready = computed(() => {
    const s = this.data.state();
    return s.status === 'ready' ? s.data : null;
  });

  protected readonly errorMessage = computed(() => {
    const s = this.data.state();
    return s.status === 'error' ? s.message : '';
  });

  protected readonly kpis = computed(() => {
    const d = this.ready();
    if (!d) return [];
    const cv = d.metrics.cv as Partial<Record<MetricKey, { mean: number; std: number }>>;
    return METRIC_ORDER.map((key) => ({
      key,
      label: METRIC_LABELS[key],
      value: d.metrics.test[key],
      cv: cv[key] ?? null,
    }));
  });

  protected readonly comparison = computed<ComparisonRow[]>(() => {
    const d = this.ready();
    if (!d) return [];
    // Comparación justa: ambos modelos con umbral 0,5 (la regresión logística no usa el umbral optimizado).
    const nn = d.metrics.test_threshold_0_5 ?? d.metrics.test;
    return METRIC_ORDER.map((key) => ({
      label: METRIC_LABELS[key],
      a: nn[key],
      b: d.metrics.baselines.logistic_regression[key],
    }));
  });

  /** Frase honesta sobre la comparación con el baseline (según ROC-AUC de test). */
  protected readonly verdict = computed(() => {
    const d = this.ready();
    if (!d) return '';
    const nn = d.metrics.test.roc_auc;
    const lr = d.metrics.baselines.logistic_regression.roc_auc;
    const diff = (nn - lr) * 100;
    if (Math.abs(diff) < 2) {
      return `Rendimiento prácticamente empatado: la diferencia de ROC-AUC en test es de ${fmtNum(Math.abs(diff), 1)} puntos, dentro del ruido esperable con ${d.metrics.n_samples.test} pacientes de test. Una regresión logística es una alternativa igual de válida y más interpretable.`;
    }
    return diff > 0
      ? `La red neuronal supera a la regresión logística en ${fmtNum(diff, 1)} puntos de ROC-AUC en test.`
      : `La regresión logística supera a la red neuronal en ${fmtNum(Math.abs(diff), 1)} puntos de ROC-AUC en test.`;
  });

  protected readonly thresholdLabel = computed(() => {
    const t = this.ready()?.metrics.threshold;
    return t === undefined
      ? 'Conjunto de test'
      : `Test · umbral optimizado ${t.toLocaleString('es-ES', { maximumFractionDigits: 3 })}`;
  });

  protected readonly fairComparison = computed(() => !!this.ready()?.metrics.test_threshold_0_5);

  protected readonly lossSeries = computed<LineSeries[]>(() => {
    const d = this.ready();
    if (!d) return [];
    const toPts = (arr: number[]) => arr.map((y, i) => ({ x: i + 1, y }));
    return [
      {
        id: 'train',
        label: 'Entrenamiento',
        points: toPts(d.history.train_loss),
        color: 'var(--app-series-1)',
      },
      {
        id: 'val',
        label: 'Validación',
        points: toPts(d.history.val_loss),
        color: 'var(--app-series-2)',
        dashed: true,
      },
    ];
  });

  protected readonly bestEpoch = computed(() => {
    const d = this.ready();
    return d ? d.history.best_epoch + 1 : 0;
  });

  protected readonly rocSeries = computed<LineSeries[]>(() => {
    const d = this.ready();
    if (!d) return [];
    const { fpr, tpr } = d.metrics.roc_curve;
    return [
      {
        id: 'roc',
        label: 'Red neuronal',
        points: fpr.map((x, i) => ({ x, y: tpr[i] })),
        color: 'var(--app-series-1)',
        area: true,
      },
    ];
  });

  protected readonly fmtEpoch = (v: number) => String(Math.round(v));
  protected readonly fmtLoss = (v: number) => fmtNum(v, 2);
  protected readonly fmtRate = (v: number) => fmtNum(v, 2);
  protected readonly epochTitle = (x: number) => `Época ${x}`;
  protected readonly rocTitle = (x: number) => `Tasa de falsos positivos ${fmtNum(x, 2)}`;
}
