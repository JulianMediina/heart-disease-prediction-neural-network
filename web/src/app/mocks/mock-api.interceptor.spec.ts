import {
  HttpClient,
  HttpErrorResponse,
  provideHttpClient,
  withInterceptors,
} from '@angular/common/http';
import { TestBed } from '@angular/core/testing';
import { firstValueFrom } from 'rxjs';

import { FEATURE_NAMES, FeatureSpec, PredictionResponse } from '../core/models/api.models';
import { MOCK_PATIENT, mockPredict, riskLevelFor } from './fixtures';
import { mockApiInterceptor } from './mock-api.interceptor';

describe('mockApiInterceptor', () => {
  let http: HttpClient;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(withInterceptors([mockApiInterceptor]))],
    });
    http = TestBed.inject(HttpClient);
  });

  it('sirve /api/features con las 13 variables en orden', async () => {
    const features = await firstValueFrom(http.get<FeatureSpec[]>('/api/features'));
    expect(features.map((f) => f.name)).toEqual([...FEATURE_NAMES]);
  });

  it('predice con un paciente válido', async () => {
    const r = await firstValueFrom(http.post<PredictionResponse>('/api/predict', MOCK_PATIENT));
    expect(r.probability).toBeGreaterThanOrEqual(0);
    expect(r.probability).toBeLessThanOrEqual(1);
    expect(r.risk_level).toBe(riskLevelFor(r.probability));
  });

  it('devuelve 422 si faltan variables', async () => {
    await expect(firstValueFrom(http.post('/api/predict', { age: 50 }))).rejects.toBeInstanceOf(
      HttpErrorResponse,
    );
  });
});

describe('fixtures', () => {
  it('riskLevelFor respeta los cortes 0,33 / 0,66', () => {
    expect(riskLevelFor(0.1)).toBe('bajo');
    expect(riskLevelFor(0.33)).toBe('moderado');
    expect(riskLevelFor(0.66)).toBe('alto');
  });

  it('mockPredict es determinista', () => {
    expect(mockPredict(MOCK_PATIENT)).toEqual(mockPredict(MOCK_PATIENT));
  });
});
