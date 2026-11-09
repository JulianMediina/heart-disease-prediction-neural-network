import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  effect,
  inject,
  input,
  signal,
} from '@angular/core';

import { prefersReducedMotion } from '../../core/services/motion';

/** Número que cuenta desde 0 hasta `value` con easing (instantáneo si se reduce el movimiento). */
@Component({
  selector: 'app-count-up',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<span class="tabular-nums">{{ prefix() }}{{ display() }}{{ suffix() }}</span>`,
})
export class CountUp {
  readonly value = input.required<number>();
  readonly decimals = input(0);
  readonly duration = input(1200);
  readonly prefix = input('');
  readonly suffix = input('');

  protected readonly display = signal('0');
  private frame = 0;

  constructor() {
    inject(DestroyRef).onDestroy(() => cancelAnimationFrame(this.frame));
    effect(() => {
      const target = this.value();
      const decimals = this.decimals();
      const duration = this.duration();
      cancelAnimationFrame(this.frame);
      if (prefersReducedMotion() || typeof requestAnimationFrame !== 'function') {
        this.display.set(this.format(target, decimals));
        return;
      }
      const start = performance.now();
      const tick = (now: number) => {
        const t = Math.min(1, (now - start) / duration);
        const eased = 1 - Math.pow(1 - t, 3);
        this.display.set(this.format(target * eased, decimals));
        if (t < 1) this.frame = requestAnimationFrame(tick);
      };
      this.frame = requestAnimationFrame(tick);
    });
  }

  private format(v: number, decimals: number): string {
    return v.toLocaleString('es-ES', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  }
}
