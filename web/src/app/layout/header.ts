import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';

import { ThemeService } from '../core/services/theme.service';
import { ApiStatusService } from '../core/services/api-status.service';
import { EcgLine } from './ecg-line';

@Component({
  selector: 'app-header',
  imports: [RouterLink, RouterLinkActive, EcgLine],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <header class="sticky top-0 z-30 border-b border-border bg-bg/85 backdrop-blur-md">
      <div class="mx-auto flex max-w-6xl items-center gap-3 px-4 py-3 sm:px-6">
        <a
          routerLink="/"
          class="flex min-w-0 items-center gap-2.5 rounded-lg"
          aria-label="CardioRed, ir al inicio"
        >
          <span
            class="relative grid size-10 shrink-0 place-items-center rounded-xl bg-linear-to-br from-flare-300 via-flare-500 to-flare-800 text-white shadow-md shadow-flare-600/30"
          >
            <svg
              viewBox="0 0 24 24"
              class="animate-heartbeat size-6"
              fill="currentColor"
              aria-hidden="true"
            >
              <path
                d="M12 21s-7.5-4.6-9.6-9.3C.9 8.3 3 4.5 6.7 4.5c2.1 0 3.6 1.1 4.3 2.4.7-1.3 2.3-2.4 4.4-2.4 3.7 0 5.8 3.8 4.3 7.2C19.5 16.4 12 21 12 21Z"
              />
            </svg>
          </span>
          <span class="min-w-0">
            <span class="block truncate text-base font-extrabold tracking-tight text-fg"
              >CardioRed</span
            >
            <span class="hidden truncate text-xs text-muted sm:block"
              >Predicción de riesgo cardíaco con redes neuronales</span
            >
          </span>
        </a>

        <nav class="ml-auto hidden items-center gap-1 md:flex" aria-label="Principal">
          @for (l of links; track l.path) {
            <a
              [routerLink]="l.path"
              routerLinkActive="!bg-flare-100 !text-flare-800 dark:!bg-flare-900/60 dark:!text-flare-100"
              [routerLinkActiveOptions]="{ exact: l.path === '/' }"
              ariaCurrentWhenActive="page"
              class="rounded-lg px-3 py-2 text-sm font-semibold text-muted transition-colors hover:bg-surface-2 hover:text-fg"
              >{{ l.label }}</a
            >
          }
        </nav>

        <span
          class="hidden items-center gap-1.5 rounded-full border border-border px-2.5 py-1 text-xs text-muted sm:inline-flex"
          [title]="status.label()"
        >
          <span class="relative inline-flex size-2.5">
            @if (status.state() === 'online') {
              <span
                class="absolute inset-0 animate-ping rounded-full bg-calma-400 opacity-60"
              ></span>
            }
            <span
              class="relative inline-block size-2.5 rounded-full"
              [class]="
                status.state() === 'online'
                  ? 'bg-calma-400'
                  : status.state() === 'checking'
                    ? 'bg-flare-200'
                    : 'bg-flare-500'
              "
            ></span>
          </span>
          {{ status.shortLabel() }}
        </span>

        <button
          type="button"
          class="ml-auto grid size-10 place-items-center rounded-xl border border-border bg-surface text-fg transition hover:bg-surface-2 md:ml-0"
          (click)="theme.toggle()"
          [attr.aria-label]="
            theme.theme() === 'dark' ? 'Activar modo claro' : 'Activar modo oscuro'
          "
          [attr.aria-pressed]="theme.theme() === 'dark'"
        >
          @if (theme.theme() === 'dark') {
            <svg
              viewBox="0 0 24 24"
              class="size-5 animate-[spin_0.5s_ease-out_1]"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              aria-hidden="true"
            >
              <circle cx="12" cy="12" r="4" />
              <path
                d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"
                stroke-linecap="round"
              />
            </svg>
          } @else {
            <svg
              viewBox="0 0 24 24"
              class="size-5 animate-[spin_0.5s_ease-out_1]"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              aria-hidden="true"
            >
              <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" stroke-linejoin="round" />
            </svg>
          }
        </button>

        <button
          type="button"
          class="grid size-10 place-items-center rounded-xl border border-border bg-surface text-fg md:hidden"
          (click)="menuOpen.set(!menuOpen())"
          [attr.aria-expanded]="menuOpen()"
          aria-controls="mobile-nav"
          aria-label="Abrir menú de navegación"
        >
          <svg
            viewBox="0 0 24 24"
            class="size-5"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            aria-hidden="true"
          >
            @if (menuOpen()) {
              <path d="M6 6l12 12M18 6 6 18" />
            } @else {
              <path d="M4 7h16M4 12h16M4 17h16" />
            }
          </svg>
        </button>
      </div>

      @if (menuOpen()) {
        <nav
          id="mobile-nav"
          class="border-t border-border px-4 pb-3 md:hidden"
          aria-label="Principal (móvil)"
          animate.enter="step-enter"
        >
          @for (l of links; track l.path) {
            <a
              [routerLink]="l.path"
              routerLinkActive="!bg-flare-100 !text-flare-800 dark:!bg-flare-900/60 dark:!text-flare-100"
              [routerLinkActiveOptions]="{ exact: l.path === '/' }"
              ariaCurrentWhenActive="page"
              class="mt-1 block rounded-lg px-3 py-2.5 text-sm font-semibold text-muted hover:bg-surface-2"
              (click)="menuOpen.set(false)"
              >{{ l.label }}</a
            >
          }
        </nav>
      }

      <app-ecg-line class="-mb-px" />
    </header>
  `,
})
export class Header {
  protected readonly theme = inject(ThemeService);
  protected readonly status = inject(ApiStatusService);
  protected readonly menuOpen = signal(false);
  protected readonly links = [
    { path: '/', label: 'Inicio' },
    { path: '/prediccion', label: 'Predicción' },
    { path: '/modelo', label: 'Modelo y datos' },
  ];
}
