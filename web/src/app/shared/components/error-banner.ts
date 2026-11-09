import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';

/** Aviso amable cuando la API no responde, con botón de reintento. */
@Component({
  selector: 'app-error-banner',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div
      role="alert"
      class="animate-fade-up flex flex-col gap-3 rounded-2xl border border-flare-300/60 bg-flare-50 p-4 text-flare-900 sm:flex-row sm:items-center dark:border-flare-700 dark:bg-flare-900/40 dark:text-flare-100"
    >
      <svg
        aria-hidden="true"
        viewBox="0 0 24 24"
        class="size-6 shrink-0 text-flare-600 dark:text-flare-200"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
      >
        <path
          d="M12 9v4m0 4h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"
          stroke-linecap="round"
          stroke-linejoin="round"
        />
      </svg>
      <div class="flex-1">
        <p class="font-semibold">{{ title() }}</p>
        <p class="text-sm opacity-90">{{ message() }}</p>
      </div>
      <button type="button" class="btn btn-primary self-start sm:self-auto" (click)="retry.emit()">
        <svg
          aria-hidden="true"
          viewBox="0 0 24 24"
          class="size-4"
          fill="none"
          stroke="currentColor"
          stroke-width="2.2"
        >
          <path
            d="M21 12a9 9 0 1 1-3-6.7L21 8M21 3v5h-5"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
        Reintentar
      </button>
    </div>
  `,
})
export class ErrorBanner {
  readonly title = input('No pudimos cargar los datos');
  readonly message = input.required<string>();
  readonly retry = output<void>();
}
