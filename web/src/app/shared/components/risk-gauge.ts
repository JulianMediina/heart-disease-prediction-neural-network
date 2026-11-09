import {
  afterNextRender,
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  signal,
} from '@angular/core';

import { CountUp } from './count-up';

/**
 * Medidor semicircular 0-100 %. Tres zonas (bajo < 33, moderado < 66, alto) y
 * una aguja que gira con transición elástica hasta la probabilidad.
 */
@Component({
  selector: 'app-risk-gauge',
  imports: [CountUp],
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block' },
  template: `
    <figure class="mx-auto w-full max-w-xs">
      <svg
        viewBox="0 0 200 120"
        class="w-full overflow-visible"
        role="img"
        [attr.aria-label]="
          'Probabilidad estimada de enfermedad cardíaca: ' + percent() + ' por ciento'
        "
      >
        <defs>
          <linearGradient id="gauge-grad" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stop-color="var(--risk-bajo)" />
            <stop offset="45%" stop-color="var(--risk-moderado)" />
            <stop offset="100%" stop-color="var(--risk-alto)" />
          </linearGradient>
        </defs>
        <path
          d="M20 100 A80 80 0 0 1 180 100"
          fill="none"
          stroke="var(--app-surface-2)"
          stroke-width="16"
          stroke-linecap="round"
        />
        <path
          d="M20 100 A80 80 0 0 1 180 100"
          fill="none"
          stroke="url(#gauge-grad)"
          stroke-width="16"
          stroke-linecap="round"
          pathLength="100"
          stroke-dasharray="100"
          [attr.stroke-dashoffset]="armed() ? 100 - percent() : 100"
          class="transition-[stroke-dashoffset] duration-[1400ms] ease-out"
        />
        <!-- Marcas de umbral -->
        @for (t of ticks; track t) {
          <line
            x1="100"
            y1="28"
            x2="100"
            y2="36"
            stroke="var(--app-muted)"
            stroke-width="1.5"
            [attr.transform]="'rotate(' + (t * 1.8 - 90) + ' 100 100)'"
          />
        }
        <g
          class="transition-transform duration-[1400ms] ease-[cubic-bezier(0.34,1.45,0.64,1)]"
          style="transform-origin: 100px 100px"
          [style.transform]="'rotate(' + angle() + 'deg)'"
        >
          <path d="M97 100 L100 30 L103 100 Z" fill="var(--app-fg)" />
        </g>
        <circle cx="100" cy="100" r="8" fill="var(--app-fg)" />
        <circle cx="100" cy="100" r="3" fill="var(--app-surface)" />
        <text x="18" y="118" text-anchor="middle" class="fill-muted text-[9px]">0%</text>
        <text x="182" y="118" text-anchor="middle" class="fill-muted text-[9px]">100%</text>
      </svg>
      <figcaption class="-mt-1 text-center">
        <span class="text-4xl font-extrabold tracking-tight text-fg"
          ><app-count-up [value]="percent()" [decimals]="1" suffix="%" [duration]="1400"
        /></span>
        <span class="block text-xs text-muted">probabilidad de enfermedad</span>
      </figcaption>
    </figure>
  `,
})
export class RiskGauge {
  /** Probabilidad 0-1. */
  readonly probability = input.required<number>();
  protected readonly ticks = [33, 66];
  protected readonly percent = computed(
    () => Math.round(Math.min(1, Math.max(0, this.probability())) * 1000) / 10,
  );
  /** -90° (0 %) … +90° (100 %). */
  protected readonly angle = computed(() => (this.armed() ? -90 + this.percent() * 1.8 : -90));
  /** Arranca en 0 y, tras el primer render, se mueve al valor real (para que la aguja se anime). */
  protected readonly armed = signal(false);

  constructor() {
    afterNextRender(() => setTimeout(() => this.armed.set(true), 40));
  }
}
