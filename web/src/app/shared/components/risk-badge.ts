import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

import { RiskLevel } from '../../core/models/api.models';

const LABELS: Record<RiskLevel, string> = {
  bajo: 'Riesgo bajo',
  moderado: 'Riesgo moderado',
  alto: 'Riesgo alto',
};

/** Insignia del nivel de riesgo: color + icono + texto (nunca sólo color). */
@Component({
  selector: 'app-risk-badge',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <span
      class="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-bold"
      [style.background]="'var(--risk-' + level() + '-bg)'"
      [style.color]="'var(--risk-' + level() + '-fg)'"
      [attr.data-level]="level()"
    >
      <svg
        aria-hidden="true"
        viewBox="0 0 24 24"
        class="size-4"
        fill="none"
        stroke="currentColor"
        stroke-width="2.4"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        @switch (level()) {
          @case ('bajo') {
            <path d="M20 6 9 17l-5-5" />
          }
          @case ('moderado') {
            <path d="M12 8v5m0 3h.01M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Z" />
          }
          @default {
            <path
              d="M12 9v4m0 4h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"
            />
          }
        }
      </svg>
      {{ label() }}
    </span>
  `,
})
export class RiskBadge {
  readonly level = input.required<RiskLevel>();
  protected readonly label = computed(() => LABELS[this.level()]);
}
