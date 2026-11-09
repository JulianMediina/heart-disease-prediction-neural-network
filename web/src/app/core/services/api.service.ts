import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { inject, Injectable, InjectionToken } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../../../environments/environment';
import {
  DatasetSummary,
  FeatureSpec,
  HealthResponse,
  MetricsResponse,
  PatientInput,
  PredictionResponse,
  TrainingHistory,
  ValidationErrorDetail,
} from '../models/api.models';

export const API_BASE_URL = new InjectionToken<string>('API_BASE_URL', {
  providedIn: 'root',
  factory: () => environment.apiBaseUrl,
});

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL).replace(/\/$/, '');

  health(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.baseUrl}/health`);
  }

  features(): Observable<FeatureSpec[]> {
    return this.http.get<FeatureSpec[]>(`${this.baseUrl}/features`);
  }

  metrics(): Observable<MetricsResponse> {
    return this.http.get<MetricsResponse>(`${this.baseUrl}/metrics`);
  }

  trainingHistory(): Observable<TrainingHistory> {
    return this.http.get<TrainingHistory>(`${this.baseUrl}/training-history`);
  }

  datasetSummary(): Observable<DatasetSummary> {
    return this.http.get<DatasetSummary>(`${this.baseUrl}/dataset/summary`);
  }

  predict(input: PatientInput): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>(`${this.baseUrl}/predict`, input);
  }
}

/** Convierte un error HTTP en un mensaje legible en español. */
export function describeApiError(err: unknown): string {
  if (err instanceof HttpErrorResponse) {
    if (err.status === 0 || err.status === 504 || err.status === 502) {
      return 'No se pudo conectar con la API. Comprueba que el servidor esté en marcha en http://localhost:8000.';
    }
    const detail = (err.error as { detail?: ValidationErrorDetail[] | string } | null)?.detail;
    if (err.status === 422) {
      if (Array.isArray(detail) && detail.length) {
        const fields = detail.map((d) => String(d.loc.at(-1))).join(', ');
        return `Datos no válidos en: ${fields}. Revisa los valores introducidos.`;
      }
      return typeof detail === 'string' ? detail : 'Los datos enviados no son válidos.';
    }
    if (err.status === 503) {
      const extra = typeof detail === 'string' ? ` (${detail})` : '';
      return `El modelo o sus artefactos todavía no están disponibles en el servidor${extra}. ¿Falta entrenarlo?`;
    }
    if (err.status === 404) {
      return 'El recurso solicitado no existe en la API.';
    }
    return `Error del servidor (${err.status}). Inténtalo de nuevo en unos segundos.`;
  }
  return 'Ha ocurrido un error inesperado.';
}
