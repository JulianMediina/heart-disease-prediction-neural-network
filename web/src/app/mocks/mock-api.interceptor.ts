import { HttpErrorResponse, HttpInterceptorFn, HttpResponse } from '@angular/common/http';
import { delay, of, throwError } from 'rxjs';

import { FEATURE_NAMES, PatientInput } from '../core/models/api.models';
import {
  MOCK_DATASET_SUMMARY,
  MOCK_FEATURES,
  MOCK_HEALTH,
  MOCK_METRICS,
  MOCK_TRAINING_HISTORY,
  mockPredict,
} from './fixtures';

const LATENCY_MS = 450;

/**
 * Responde a las peticiones `/api/*` con los fixtures locales.
 * Se registra sólo cuando `environment.useMocks` es true.
 */
export const mockApiInterceptor: HttpInterceptorFn = (req, next) => {
  const match = req.url.match(/\/api(\/[^?]*)/);
  if (!match) return next(req);
  const path = match[1];

  const ok = (body: unknown) =>
    of(new HttpResponse({ status: 200, body, url: req.url })).pipe(delay(LATENCY_MS));

  if (req.method === 'GET') {
    switch (path) {
      case '/health':
        return ok(MOCK_HEALTH);
      case '/features':
        return ok(MOCK_FEATURES);
      case '/metrics':
        return ok(MOCK_METRICS);
      case '/training-history':
        return ok(MOCK_TRAINING_HISTORY);
      case '/dataset/summary':
        return ok(MOCK_DATASET_SUMMARY);
    }
  }

  if (req.method === 'POST' && path === '/predict') {
    const body = (req.body ?? {}) as Partial<PatientInput>;
    const errors = FEATURE_NAMES.filter(
      (n) => typeof body[n] !== 'number' || !Number.isFinite(body[n]),
    ).map((n) => ({
      loc: ['body', n],
      msg: 'Field required',
      type: 'missing',
    }));
    if (errors.length) {
      return throwError(
        () =>
          new HttpErrorResponse({
            status: 422,
            statusText: 'Unprocessable Entity',
            error: { detail: errors },
            url: req.url,
          }),
      ).pipe(delay(LATENCY_MS));
    }
    return ok(mockPredict(body as PatientInput));
  }

  return throwError(
    () => new HttpErrorResponse({ status: 404, statusText: 'Not Found', url: req.url }),
  );
};
