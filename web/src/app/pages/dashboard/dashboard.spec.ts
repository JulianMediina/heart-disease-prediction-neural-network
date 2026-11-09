import {
  HttpErrorResponse,
  HttpInterceptorFn,
  provideHttpClient,
  withInterceptors,
} from '@angular/common/http';
import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { throwError } from 'rxjs';
import { vi } from 'vitest';

import { mockApiInterceptor } from '../../mocks/mock-api.interceptor';
import { Dashboard } from './dashboard';

async function render(interceptor: HttpInterceptorFn) {
  await TestBed.configureTestingModule({
    imports: [Dashboard],
    providers: [provideRouter([]), provideHttpClient(withInterceptors([interceptor]))],
  }).compileComponents();
  const fixture = TestBed.createComponent(Dashboard);
  fixture.detectChanges();
  await vi.advanceTimersByTimeAsync(600);
  fixture.detectChanges();
  await vi.advanceTimersByTimeAsync(0);
  fixture.detectChanges();
  return fixture;
}

describe('Dashboard', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('muestra KPIs, matriz de confusión y una comparación honesta', async () => {
    const fixture = await render(mockApiInterceptor);
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelectorAll('app-kpi-card')).toHaveLength(5);
    expect(el.querySelectorAll('app-confusion-matrix [role="cell"]')).toHaveLength(4);
    expect(el.textContent).toContain('Regresión logística (umbral 0,5)');
    expect(el.textContent).toContain('prácticamente empatado');
    expect(el.textContent).toContain('umbral optimizado');
    expect(el.textContent).toContain('tras eliminar duplicados');
  });

  it('muestra un aviso con reintento si la API no responde', async () => {
    const down: HttpInterceptorFn = () => throwError(() => new HttpErrorResponse({ status: 0 }));
    const fixture = await render(down);
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('[role="alert"]')?.textContent).toContain('No se pudo conectar');
    expect(el.querySelector('app-error-banner button')?.textContent).toContain('Reintentar');
  });
});
