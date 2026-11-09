import { registerLocaleData } from '@angular/common';
import { provideHttpClient, withInterceptors } from '@angular/common/http';
import localeEs from '@angular/common/locales/es';
import { LOCALE_ID } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { vi } from 'vitest';

import { MOCK_PATIENT } from '../../mocks/fixtures';
import { mockApiInterceptor } from '../../mocks/mock-api.interceptor';
import { Prediction } from './prediction';

registerLocaleData(localeEs);

describe('Prediction', () => {
  let fixture: ComponentFixture<Prediction>;
  let el: HTMLElement;

  async function settle(ms = 500): Promise<void> {
    await vi.advanceTimersByTimeAsync(ms);
    fixture.detectChanges();
    await vi.advanceTimersByTimeAsync(0);
    fixture.detectChanges();
  }

  const legend = () => el.querySelector('legend')?.textContent ?? '';
  const submit = () => (el.querySelector('button[type="submit"]') as HTMLButtonElement).click();

  beforeEach(async () => {
    vi.useFakeTimers();
    await TestBed.configureTestingModule({
      imports: [Prediction],
      providers: [
        provideHttpClient(withInterceptors([mockApiInterceptor])),
        { provide: LOCALE_ID, useValue: 'es' },
      ],
    }).compileComponents();
    fixture = TestBed.createComponent(Prediction);
    el = fixture.nativeElement;
    fixture.detectChanges();
    await settle();
  });

  afterEach(() => vi.useRealTimers());

  it('muestra el aviso médico y la nota sobre la etiqueta', () => {
    expect(el.textContent).toContain('no sustituye un diagnóstico médico');
    expect(el.textContent).toContain('probabilidad de enfermedad cardíaca');
    expect(el.textContent).not.toContain('clase 1 del dataset');
  });

  it('genera el primer paso desde /api/features', () => {
    expect(legend()).toContain('Datos personales');
    expect(el.querySelector('#f-age')).not.toBeNull();
    expect(el.querySelectorAll('input[type="radio"][name="sex"]')).toHaveLength(2);
  });

  it('no avanza si hay campos obligatorios vacíos', async () => {
    submit();
    await settle(50);
    expect(legend()).toContain('Datos personales');
    expect(el.textContent).toContain('Selecciona una opción.');
  });

  it('valida min/max', () => {
    const age = fixture.componentInstance.control('age');
    age.setValue(200);
    expect(age.hasError('max')).toBe(true);
    age.setValue(10);
    expect(age.hasError('min')).toBe(true);
  });

  it('completa los pasos y muestra el resultado con medidor, banda y umbral', async () => {
    const comp = fixture.componentInstance;
    for (const [k, v] of Object.entries(MOCK_PATIENT)) comp.control(k).setValue(v);
    submit();
    await settle(50);
    expect(legend()).toContain('Síntomas');
    submit();
    await settle(50);
    expect(legend()).toContain('Pruebas clínicas');
    submit();
    await settle(600);
    const live = el.querySelector('[aria-live="polite"]');
    expect(live?.querySelector('app-risk-gauge')).not.toBeNull();
    expect(live?.querySelector('app-risk-badge')?.textContent).toMatch(
      /Riesgo (bajo|moderado|alto)/,
    );
    expect(live?.textContent).toContain('Banda');
    // El umbral viene de metrics.threshold (0,345 en los fixtures), no está fijado en el código.
    expect(live?.textContent).toContain('0,345');
  });
});
