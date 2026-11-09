import { ChangeDetectionStrategy, Component, DestroyRef, inject, signal } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';

import { ApiStatusService } from '../core/services/api-status.service';
import { ThemeService } from '../core/services/theme.service';
import { EcgLine } from './ecg-line';

/** Barra superior estilo monitor: identificación, pestañas, estado de señal, reloj y tema. */
@Component({
  selector: 'app-header',
  imports: [RouterLink, RouterLinkActive, EcgLine],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <header class="sticky top-0 z-30 border-b border-border bg-bg/90 backdrop-blur-md">
      <div class="mx-auto flex max-w-6xl items-center gap-3 px-4 py-2.5 sm:px-6">
        <a routerLink="/" class="flex min-w-0 items-center gap-3 rounded" aria-label="CardioRed, ir al inicio">
          <span
            class="relative grid size-10 shrink-0 place-items-center rounded-md border border-border-strong bg-surface text-hg-sage dark:text-hg-sage"
          >
            <svg viewBox="0 0 24 24" class="animate-heartbeat glow size-6" fill="currentColor" aria-hidden="true">
              <path
                d="M12 21s-7.5-4.6-9.6-9.3C.9 8.3 3 4.5 6.7 4.5c2.1 0 3.6 1.1 4.3 2.4.7-1.3 2.3-2.4 4.4-2.4 3.7 0 5.8 3.8 4.3 7.2C19.5 16.4 12 21 12 21Z"
              />
            </svg>
          </span>
          <span class="min-w-0">
            <span class="block truncate text-lg leading-tight font-bold tracking-wide text-fg">CardioRed</span>
            <span class="channel-label hidden truncate !text-[10px] sm:block">Monitor · red neuronal</span>
          </span>
        </a>

        <nav class="ml-auto hidden items-center gap-1 md:flex" aria-label="Principal">
          @for (l of links; track l.path; let i = $index) {
            <a
              [routerLink]="l.path"
              routerLinkActive="!border-hg-sage !text-fg bg-surface-2"
              [routerLinkActiveOptions]="{ exact: l.path === '/' }"
              ariaCurrentWhenActive="page"
              class="rounded-sm border-b-2 border-transparent px-3 py-2 font-mono text-xs font-semibold tracking-widest text-muted uppercase transition-colors hover:text-fg"
              ><span class="text-muted/70" aria-hidden="true">{{ i + 1 }} </span>{{ l.label }}</a
            >
          }
        </nav>

        <span
          class="hidden items-center gap-2 rounded-sm border border-border px-2.5 py-1 font-mono text-[11px] tracking-wider text-muted uppercase lg:inline-flex"
          [title]="status.label()"
        >
          <span
            class="inline-block size-2 rounded-full"
            [class]="
              status.state() === 'online'
                ? 'animate-blink bg-hg-sage shadow-[0_0_8px_var(--color-hg-sage)]'
                : status.state() === 'checking'
                  ? 'bg-hg-mint'
                  : 'animate-blink bg-hg-alert'
            "
          ></span>
          {{ status.shortLabel() }}
        </span>
        <span class="vital hidden text-sm text-fg sm:inline" aria-hidden="true">{{ clock() }}</span>

        <button
          type="button"
          class="ml-auto grid size-10 place-items-center rounded-md border border-border bg-surface text-fg transition hover:border-border-strong md:ml-0"
          (click)="theme.toggle()"
          [attr.aria-label]="theme.theme() === 'dark' ? 'Activar modo claro' : 'Activar modo oscuro'"
          [attr.aria-pressed]="theme.theme() === 'dark'"
        >
          @if (theme.theme() === 'dark') {
            <svg viewBox="0 0 24 24" class="size-5" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <circle cx="12" cy="12" r="4" />
              <path
                d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"
                stroke-linecap="round"
              />
            </svg>
          } @else {
            <svg viewBox="0 0 24 24" class="size-5" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" stroke-linejoin="round" />
            </svg>
          }
        </button>

        <button
          type="button"
          class="grid size-10 place-items-center rounded-md border border-border bg-surface text-fg md:hidden"
          (click)="menuOpen.set(!menuOpen())"
          [attr.aria-expanded]="menuOpen()"
          aria-controls="mobile-nav"
          aria-label="Abrir menú de navegación"
        >
          <svg viewBox="0 0 24 24" class="size-5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">
            @if (menuOpen()) {
              <path d="M6 6l12 12M18 6 6 18" />
            } @else {
              <path d="M4 7h16M4 12h16M4 17h16" />
            }
          </svg>
        </button>
      </div>

      @if (menuOpen()) {
        <nav id="mobile-nav" class="border-t border-border px-4 pb-3 md:hidden" aria-label="Principal (móvil)" animate.enter="step-enter">
          @for (l of links; track l.path) {
            <a
              [routerLink]="l.path"
              routerLinkActive="!text-fg bg-surface-2"
              [routerLinkActiveOptions]="{ exact: l.path === '/' }"
              ariaCurrentWhenActive="page"
              class="mt-1 block rounded-sm px-3 py-2.5 font-mono text-xs font-semibold tracking-widest text-muted uppercase"
              (click)="menuOpen.set(false)"
              >{{ l.label }}</a
            >
          }
        </nav>
      }

      <app-ecg-line class="-mb-px h-7" [beats]="6" [speed]="3.4" />
    </header>
  `,
})
export class Header {
  protected readonly theme = inject(ThemeService);
  protected readonly status = inject(ApiStatusService);
  protected readonly menuOpen = signal(false);
  protected readonly clock = signal(formatTime(new Date()));
  protected readonly links = [
    { path: '/', label: 'Inicio' },
    { path: '/prediccion', label: 'Predicción' },
    { path: '/modelo', label: 'Modelo y datos' },
  ];

  constructor() {
    const id = setInterval(() => this.clock.set(formatTime(new Date())), 1000);
    inject(DestroyRef).onDestroy(() => clearInterval(id));
  }
}

function formatTime(d: Date): string {
  return d.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
