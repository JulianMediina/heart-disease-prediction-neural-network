import { DestroyRef, inject, signal, Signal } from '@angular/core';
import { Observable, Subscription } from 'rxjs';

import { describeApiError } from './api.service';

export type Loadable<T> =
  { status: 'loading' } | { status: 'error'; message: string } | { status: 'ready'; data: T };

export interface LoadableRef<T> {
  state: Signal<Loadable<T>>;
  reload: () => void;
}

/**
 * Envuelve una petición en un signal con estados loading / error / ready
 * y expone `reload()` para reintentar. Debe llamarse en contexto de inyección.
 */
export function loadable<T>(source: () => Observable<T>): LoadableRef<T> {
  const state = signal<Loadable<T>>({ status: 'loading' });
  let sub: Subscription | undefined;
  const reload = () => {
    sub?.unsubscribe();
    state.set({ status: 'loading' });
    sub = source().subscribe({
      next: (data) => state.set({ status: 'ready', data }),
      error: (err: unknown) => state.set({ status: 'error', message: describeApiError(err) }),
    });
  };
  inject(DestroyRef).onDestroy(() => sub?.unsubscribe());
  reload();
  return { state: state.asReadonly(), reload };
}
