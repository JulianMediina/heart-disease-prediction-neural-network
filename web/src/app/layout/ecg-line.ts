import { ChangeDetectionStrategy, Component } from '@angular/core';

/** Línea de electrocardiograma que se dibuja en bucle bajo la cabecera (decorativa). */
@Component({
  selector: 'app-ecg-line',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block pointer-events-none', 'aria-hidden': 'true' },
  template: `
    <svg viewBox="0 0 600 40" preserveAspectRatio="none" class="h-8 w-full">
      <defs>
        <linearGradient id="ecg-grad" x1="0" x2="1">
          <stop offset="0%" stop-color="var(--color-flare-200)" stop-opacity="0" />
          <stop offset="20%" stop-color="var(--color-flare-300)" />
          <stop offset="55%" stop-color="var(--color-flare-500)" />
          <stop offset="100%" stop-color="var(--color-flare-700)" stop-opacity="0.2" />
        </linearGradient>
      </defs>
      <path [attr.d]="path" fill="none" stroke="var(--app-border)" stroke-width="1.5" />
      <path
        [attr.d]="path"
        fill="none"
        stroke="url(#ecg-grad)"
        stroke-width="2.2"
        stroke-linejoin="round"
        stroke-linecap="round"
        pathLength="1"
        stroke-dasharray="0.35 0.65"
        class="animate-ecg"
      />
    </svg>
  `,
})
export class EcgLine {
  /** Tres latidos (onda P, complejo QRS, onda T) sobre una línea base. */
  protected readonly path = [0, 200, 400]
    .map(
      (o) =>
        `M${o},22 L${o + 60},22 Q${o + 70},14 ${o + 80},22 L${o + 92},22 L${o + 98},28 L${o + 106},2 L${o + 114},36 L${o + 120},22 ` +
        `L${o + 135},22 Q${o + 152},10 ${o + 168},22 L${o + 200},22`,
    )
    .join(' ');
}
