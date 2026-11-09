import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import {
  MOCK_FEATURES,
  MOCK_HEALTH,
  MOCK_METRICS,
  MOCK_PATIENT,
  MOCK_TRAINING_HISTORY,
} from '../../mocks/fixtures';
import { API_BASE_URL, ApiService, describeApiError } from './api.service';

describe('ApiService', () => {
  let api: ApiService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: API_BASE_URL, useValue: '/api/' },
      ],
    });
    api = TestBed.inject(ApiService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('GET /api/health', () => {
    let body: unknown;
    api.health().subscribe((h) => (body = h));
    const req = http.expectOne('/api/health');
    expect(req.request.method).toBe('GET');
    req.flush(MOCK_HEALTH);
    expect(body).toEqual(MOCK_HEALTH);
  });

  it('GET /api/features', () => {
    let body: unknown;
    api.features().subscribe((b) => (body = b));
    http.expectOne('/api/features').flush(MOCK_FEATURES);
    expect(body).toEqual(MOCK_FEATURES);
  });

  it('GET /api/metrics', () => {
    let body: unknown;
    api.metrics().subscribe((b) => (body = b));
    http.expectOne('/api/metrics').flush(MOCK_METRICS);
    expect(body).toEqual(MOCK_METRICS);
  });

  it('GET /api/training-history', () => {
    let body: unknown;
    api.trainingHistory().subscribe((b) => (body = b));
    http.expectOne('/api/training-history').flush(MOCK_TRAINING_HISTORY);
    expect(body).toEqual(MOCK_TRAINING_HISTORY);
  });

  it('GET /api/dataset/summary', () => {
    api.datasetSummary().subscribe();
    expect(http.expectOne('/api/dataset/summary').request.method).toBe('GET');
  });

  it('POST /api/predict envía las 13 variables', () => {
    let body: unknown;
    api.predict(MOCK_PATIENT).subscribe((r) => (body = r));
    const req = http.expectOne('/api/predict');
    expect(req.request.method).toBe('POST');
    expect(Object.keys(req.request.body)).toHaveLength(13);
    req.flush({ probability: 0.72, prediction: 1, risk_level: 'alto' });
    expect(body).toEqual({ probability: 0.72, prediction: 1, risk_level: 'alto' });
  });

  it('propaga el 422 con un mensaje legible', () => {
    let message = '';
    api.predict(MOCK_PATIENT).subscribe({ error: (e: unknown) => (message = describeApiError(e)) });
    http
      .expectOne('/api/predict')
      .flush(
        { detail: [{ loc: ['body', 'chol'], msg: 'too big', type: 'less_than_equal' }] },
        { status: 422, statusText: 'Unprocessable Entity' },
      );
    expect(message).toContain('chol');
  });
});

describe('describeApiError', () => {
  it('servidor caído (status 0)', () => {
    expect(describeApiError(new HttpErrorResponse({ status: 0 }))).toContain('No se pudo conectar');
  });

  it('503 incluye el detalle del servidor', () => {
    const msg = describeApiError(
      new HttpErrorResponse({ status: 503, error: { detail: 'falta model.pt' } }),
    );
    expect(msg).toContain('falta model.pt');
  });

  it('error desconocido', () => {
    expect(describeApiError(new Error('x'))).toBe('Ha ocurrido un error inesperado.');
  });
});
