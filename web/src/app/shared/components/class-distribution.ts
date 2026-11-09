import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

import { CountUp } from './count-up';

/** Distribución de clases como dos barras verticales con recuento y porcentaje. */
@Component({
  selector: 'app-class-distribution',
  imports: [CountUp, DecimalPipe],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div
      class="flex h-56 items-end justify-center gap-6 border-b border-border px-2 sm:gap-12"
      role="list"
      aria-label="Distribución de clases del dataset"
    >
      @for (b of bars(); track b.key; let i = $index) {
        <div
          role="listitem"
          class="flex h-full w-24 flex-col items-center justify-end sm:w-28"
          [attr.aria-label]="b.label + ': ' + b.count + ' pacientes'"
        >
          <p class="mb-1 text-lg font-bold text-fg"><app-count-up [value]="b.count" /></p>
          <p class="mb-2 text-xs text-muted tabular-nums">{{ b.pct | number: '1.1-1' }}%</p>
          <div
            class="w-full origin-bottom animate-[grow-y_1s_cubic-bezier(0.22,1,0.36,1)_both] rounded-t-lg"
            [style.height.%]="b.height"
            [style.background]="b.color"
            [style.animation-delay.ms]="i * 120"
          ></div>
        </div>
      }
    </div>
    <div class="mt-2 flex justify-center gap-6 sm:gap-12">
      @for (b of bars(); track b.key) {
        <p class="w-24 text-center text-xs font-medium text-muted sm:w-28">{{ b.label }}</p>
      }
    </div>
  `,
  styles: `
    @keyframes grow-y {
      from {
        transform: scaleY(0);
      }
    }
  `,
})
export class ClassDistribution {
  readonly distribution = input.required<Record<string, number>>();

  protected readonly bars = computed(() => {
    const d = this.distribution();
    const n0 = d['0'] ?? 0;
    const n1 = d['1'] ?? 0;
    const total = n0 + n1 || 1;
    const max = Math.max(n0, n1, 1);
    return [
      {
        key: '0',
        label: 'Sin enfermedad (0)',
        count: n0,
        pct: (n0 / total) * 100,
        height: (n0 / max) * 70,
        color: 'var(--app-series-2)',
      },
      {
        key: '1',
        label: 'Con enfermedad (1)',
        count: n1,
        pct: (n1 / total) * 100,
        height: (n1 / max) * 70,
        color: 'var(--app-series-1)',
      },
    ];
  });
}
