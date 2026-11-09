import { registerLocaleData } from '@angular/common';
import { provideHttpClient, withFetch, withInterceptors } from '@angular/common/http';
import localeEs from '@angular/common/locales/es';
import { ApplicationConfig, LOCALE_ID, provideBrowserGlobalErrorListeners } from '@angular/core';
import {
  provideRouter,
  withComponentInputBinding,
  withInMemoryScrolling,
  withViewTransitions,
} from '@angular/router';

import { environment } from '../environments/environment';
import { routes } from './app.routes';
import { mockApiInterceptor } from './mocks/mock-api.interceptor';

registerLocaleData(localeEs);

const reducedMotion =
  typeof window !== 'undefined' &&
  typeof window.matchMedia === 'function' &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    { provide: LOCALE_ID, useValue: 'es' },
    provideRouter(
      routes,
      withComponentInputBinding(),
      withInMemoryScrolling({ scrollPositionRestoration: 'top' }),
      // Transiciones de ruta animadas (View Transitions API); se omiten con movimiento reducido.
      ...(reducedMotion ? [] : [withViewTransitions({ skipInitialTransition: true })]),
    ),
    provideHttpClient(
      withFetch(),
      withInterceptors(environment.useMocks ? [mockApiInterceptor] : []),
    ),
  ],
};
