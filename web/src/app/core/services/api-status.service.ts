import { computed, inject, Injectable, signal } from '@angular/core';

import { environment } from '../../../environments/environment';
import { HealthResponse } from '../models/api.models';
import { ApiService } from './api.service';

export type ApiState = 'checking' | 'online' | 'no-model' | 'offline';

/** Estado de conexión con la API (GET /api/health). */
@Injectable({ providedIn: 'root' })
export class ApiStatusService {
  private readonly api = inject(ApiService);
  readonly health = signal<HealthResponse | null>(null);
  readonly state = signal<ApiState>('checking');
  readonly usingMocks = environment.useMocks;

  readonly shortLabel = computed(() => {
    switch (this.state()) {
      case 'online':
        return this.usingMocks ? 'Datos de ejemplo' : 'API conectada';
      case 'no-model':
        return 'Modelo no cargado';
      case 'offline':
        return 'API sin conexión';
      default:
        return 'Conectando…';
    }
  });

  readonly label = computed(() => {
    const h = this.health();
    if (this.state() === 'online' && h) return `Modelo ${h.model_version ?? 'sin versión'} cargado`;
    return this.shortLabel();
  });

  check(): void {
    this.state.set('checking');
    this.api.health().subscribe({
      next: (h) => {
        this.health.set(h);
        this.state.set(h.model_loaded ? 'online' : 'no-model');
      },
      error: () => {
        this.health.set(null);
        this.state.set('offline');
      },
    });
  }
}
