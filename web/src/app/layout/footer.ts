import { ChangeDetectionStrategy, Component, inject } from '@angular/core';

import { ApiStatusService } from '../core/services/api-status.service';

@Component({
  selector: 'app-footer',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <footer class="mt-16 border-t border-border bg-surface-2/60">
      <div
        class="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-6 text-xs text-muted sm:flex-row sm:items-center sm:justify-between sm:px-6"
      >
        <p>
          <strong class="font-semibold text-fg">Aviso:</strong>
          herramienta educativa. Los resultados no sustituyen el diagnóstico de un profesional
          sanitario.
        </p>
        <p class="shrink-0">
          Dataset Heart Disease (Cleveland/UCI) · PyTorch + FastAPI + Angular
          @if (status.usingMocks) {
            · <span class="font-semibold text-link">modo datos de ejemplo</span>
          }
        </p>
      </div>
    </footer>
  `,
})
export class Footer {
  protected readonly status = inject(ApiStatusService);
}
