import { ChangeDetectionStrategy, Component, inject, OnInit } from '@angular/core';
import { RouterOutlet } from '@angular/router';

import { ApiStatusService } from './core/services/api-status.service';
import { Footer } from './layout/footer';
import { Header } from './layout/header';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, Header, Footer],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <a
      href="#contenido"
      class="sr-only z-50 rounded-lg bg-primary px-4 py-2 text-on-primary focus:not-sr-only focus:fixed focus:left-4 focus:top-4"
      >Saltar al contenido</a
    >
    <div class="flex min-h-dvh flex-col">
      <app-header />
      @if (status.state() === 'offline') {
        <div class="mx-auto mt-4 w-full max-w-6xl px-4 sm:px-6" role="status">
          <p
            class="flex flex-wrap items-center gap-2 rounded-xl border border-border bg-surface-2 px-4 py-2 text-sm text-fg"
          >
            <span class="inline-block size-2 rounded-full bg-flare-500" aria-hidden="true"></span>
            La API no responde en este momento. Arranca el servidor en
            <code class="text-link">localhost:8000</code> o usa
            <code class="text-link">npm run start:mocks</code>.
            <button
              type="button"
              class="ml-auto font-semibold text-link underline-offset-2 hover:underline"
              (click)="status.check()"
            >
              Reintentar conexión
            </button>
          </p>
        </div>
      }
      <main id="contenido" class="flex-1" tabindex="-1">
        <router-outlet />
      </main>
      <app-footer />
    </div>
  `,
})
export class App implements OnInit {
  protected readonly status = inject(ApiStatusService);

  ngOnInit(): void {
    this.status.check();
  }
}
