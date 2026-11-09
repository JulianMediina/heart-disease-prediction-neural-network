import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { App } from './app';
import { mockApiInterceptor } from './mocks/mock-api.interceptor';

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [provideRouter([]), provideHttpClient(withInterceptors([mockApiInterceptor]))],
    }).compileComponents();
  });

  it('renderiza cabecera con navegación, conmutador de tema y pie', () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('app-header nav')?.textContent).toContain('Predicción');
    expect(el.querySelector('button[aria-pressed]')).not.toBeNull();
    expect(el.querySelector('app-footer')?.textContent).toContain('no sustituyen el diagnóstico');
  });

  it('el conmutador alterna la clase dark en <html>', () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    TestBed.tick();
    const before = document.documentElement.classList.contains('dark');
    (fixture.nativeElement.querySelector('button[aria-pressed]') as HTMLButtonElement).click();
    TestBed.tick();
    expect(document.documentElement.classList.contains('dark')).toBe(!before);
  });
});
