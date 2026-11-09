import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { catchError, forkJoin, of } from 'rxjs';

import { LABEL_NOTE } from '../../core/constants';
import { DatasetSummary, FeatureSpec, MetricsResponse } from '../../core/models/api.models';
import { ApiService } from '../../core/services/api.service';
import { ApiStatusService } from '../../core/services/api-status.service';
import { loadable } from '../../core/services/loadable';
import { ErrorBanner } from '../../shared/components/error-banner';
import { Skeleton } from '../../shared/components/skeleton';

interface ModelInfoData {
  features: FeatureSpec[];
  summary: DatasetSummary;
  metrics: MetricsResponse | null;
}

/** Capas de la arquitectura (configuración del entrenamiento v2.0.0, ver artifacts/metadata.json). */
const LAYERS = [
  {
    name: 'Entrada',
    size: 27,
    detail:
      '13 variables → 27 columnas (one-hot de cp, restecg, slp, caa, thall + estandarización)',
  },
  { name: 'Oculta 1', size: 64, detail: 'Lineal + ReLU + Dropout 0,3' },
  { name: 'Oculta 2', size: 32, detail: 'Lineal + ReLU + Dropout 0,3' },
  { name: 'Salida', size: 1, detail: 'Logit → sigmoide = P(enfermedad)' },
];

@Component({
  selector: 'app-model-info',
  imports: [DecimalPipe, ErrorBanner, Skeleton],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './model-info.html',
})
export class ModelInfo {
  private readonly api = inject(ApiService);
  protected readonly status = inject(ApiStatusService);
  protected readonly layers = LAYERS;

  protected readonly data = loadable<ModelInfoData>(() =>
    forkJoin({
      features: this.api.features(),
      summary: this.api.datasetSummary(),
      metrics: this.api.metrics().pipe(catchError(() => of(null))),
    }),
  );

  protected readonly ready = computed(() => {
    const s = this.data.state();
    return s.status === 'ready' ? s.data : null;
  });

  protected readonly errorMessage = computed(() => {
    const s = this.data.state();
    return s.status === 'error' ? s.message : '';
  });

  protected readonly labelNote = LABEL_NOTE;
  protected readonly serverNote = computed(() => this.status.health()?.target_note || null);

  /** Nodos dibujados por capa (se recortan las capas grandes). */
  protected readonly diagram = computed(() => {
    const shown = [6, 8, 6, 1];
    const w = 560;
    const h = 220;
    const cols = LAYERS.map((l, i) => {
      const x = 50 + (i * (w - 100)) / (LAYERS.length - 1);
      const n = shown[i];
      const nodes = Array.from({ length: n }, (_, j) => ({
        x,
        y: n === 1 ? h / 2 : 24 + (j * (h - 48)) / (n - 1),
      }));
      return { ...l, x, nodes };
    });
    const edges = cols
      .slice(0, -1)
      .flatMap((c, i) =>
        c.nodes.flatMap((a) =>
          cols[i + 1].nodes.map((b) => ({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, col: i })),
        ),
      );
    return { w, h, cols, edges };
  });

  protected readonly rows = computed(() => {
    const d = this.ready();
    if (!d) return [];
    return d.features.map((f) => ({ spec: f, stats: d.summary.features[f.name] ?? null }));
  });

  protected count(dist: Record<string, number> | undefined, key: string): number {
    return dist?.[key] ?? 0;
  }
}
