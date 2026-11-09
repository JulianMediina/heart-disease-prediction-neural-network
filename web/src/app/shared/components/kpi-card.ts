import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, input } from '@angular/core';

import { CountUp } from './count-up';

/** Tarjeta de métrica: valor en % con count-up, media ± desv. de CV opcional y barra animada. */
@Component({
  selector: 'app-kpi-card',
  imports: [CountUp, DecimalPipe],
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block' },
  template: `
    <article
      class="card relative h-full overflow-hidden transition-transform duration-300 hover:-translate-y-0.5"
    >
      <h3 class="text-sm font-medium text-muted">{{ label() }}</h3>
      <p class="mt-2 text-3xl font-extrabold tracking-tight text-fg sm:text-4xl">
        <app-count-up [value]="value() * 100" [decimals]="1" suffix="%" />
      </p>
      <p class="mt-1 text-xs text-muted">{{ hint() }}</p>
      @if (cv(); as c) {
        <p class="mt-0.5 text-xs text-muted">
          CV:
          <span class="font-semibold text-fg tabular-nums"
            >{{ c.mean * 100 | number: '1.1-1' }}% ± {{ c.std * 100 | number: '1.1-1' }}</span
          >
        </p>
      }
      <div class="mt-4 h-1.5 overflow-hidden rounded-full bg-surface-2" aria-hidden="true">
        <div
          class="animate-grow h-full origin-left rounded-full bg-linear-to-r from-flare-200 via-flare-400 to-flare-700"
          [style.width.%]="value() * 100"
        ></div>
      </div>
    </article>
  `,
})
export class KpiCard {
  readonly label = input.required<string>();
  /** Valor 0-1. */
  readonly value = input.required<number>();
  readonly cv = input<{ mean: number; std: number } | null>(null);
  readonly hint = input('Conjunto de test');
}
