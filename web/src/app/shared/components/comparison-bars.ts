import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, input } from '@angular/core';

export interface ComparisonRow {
  label: string;
  a: number;
  b: number;
}

/** Barras horizontales agrupadas (0-1) para comparar dos modelos métrica a métrica. */
@Component({
  selector: 'app-comparison-bars',
  imports: [DecimalPipe],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <ul class="mb-4 flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted">
      <li class="flex items-center gap-2">
        <span class="inline-block h-2.5 w-4 rounded-sm bg-series-1"></span>{{ labelA() }}
      </li>
      <li class="flex items-center gap-2">
        <span
          class="inline-block h-2.5 w-4 rounded-sm bg-series-2 bg-[repeating-linear-gradient(135deg,transparent_0_3px,rgb(255_255_255/0.35)_3px_5px)]"
        ></span
        >{{ labelB() }}
      </li>
    </ul>
    <dl class="space-y-4">
      @for (r of rows(); track r.label; let i = $index) {
        <div class="stagger-in" [style.--i]="i">
          <dt class="mb-1.5 text-sm font-medium text-fg">{{ r.label }}</dt>
          <dd class="space-y-1">
            <div
              class="flex items-center gap-2"
              [attr.title]="labelA() + ': ' + (r.a * 100 | number: '1.1-1') + '%'"
            >
              <div class="h-3 flex-1 overflow-hidden rounded-full bg-surface-2">
                <div
                  class="animate-grow h-full origin-left rounded-full bg-series-1"
                  [style.width.%]="r.a * 100"
                  [style.animation-delay.ms]="i * 80"
                ></div>
              </div>
              <span class="w-14 text-right text-xs font-semibold tabular-nums text-fg">
                <span class="sr-only">{{ labelA() }}: </span>{{ r.a * 100 | number: '1.1-1' }}%
              </span>
            </div>
            <div
              class="flex items-center gap-2"
              [attr.title]="labelB() + ': ' + (r.b * 100 | number: '1.1-1') + '%'"
            >
              <div class="h-3 flex-1 overflow-hidden rounded-full bg-surface-2">
                <div
                  class="animate-grow h-full origin-left rounded-full bg-series-2 bg-[repeating-linear-gradient(135deg,transparent_0_3px,rgb(255_255_255/0.35)_3px_5px)]"
                  [style.width.%]="r.b * 100"
                  [style.animation-delay.ms]="i * 80 + 60"
                ></div>
              </div>
              <span class="w-14 text-right text-xs tabular-nums text-muted">
                <span class="sr-only">{{ labelB() }}: </span>{{ r.b * 100 | number: '1.1-1' }}%
              </span>
            </div>
          </dd>
        </div>
      }
    </dl>
  `,
})
export class ComparisonBars {
  readonly rows = input.required<ComparisonRow[]>();
  readonly labelA = input('Red neuronal');
  readonly labelB = input('Regresión logística');
}
