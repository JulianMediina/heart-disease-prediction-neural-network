import { DecimalPipe } from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  inject,
  signal,
} from '@angular/core';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';

import { DEFAULT_TARGET_NOTE } from '../../core/constants';
import {
  FeatureName,
  FeatureSpec,
  PatientInput,
  PredictionResponse,
} from '../../core/models/api.models';
import { ApiService, describeApiError } from '../../core/services/api.service';
import { ApiStatusService } from '../../core/services/api-status.service';
import { loadable } from '../../core/services/loadable';
import { ErrorBanner } from '../../shared/components/error-banner';
import { RiskBadge } from '../../shared/components/risk-badge';
import { RiskGauge } from '../../shared/components/risk-gauge';
import { Skeleton } from '../../shared/components/skeleton';

export interface FormStep {
  id: string;
  title: string;
  description: string;
  fields: FeatureName[];
}

/** Agrupación de las 13 variables en pasos. Variables desconocidas van al último paso. */
export const STEPS: FormStep[] = [
  {
    id: 'personal',
    title: 'Datos personales',
    description: 'Edad, sexo y mediciones básicas en reposo.',
    fields: ['age', 'sex', 'trtbps', 'chol', 'fbs'],
  },
  {
    id: 'sintomas',
    title: 'Síntomas',
    description: 'Dolor torácico y respuesta al ejercicio.',
    fields: ['cp', 'exng'],
  },
  {
    id: 'pruebas',
    title: 'Pruebas clínicas',
    description: 'Electrocardiograma, prueba de esfuerzo y estudios de imagen.',
    fields: ['restecg', 'thalachh', 'oldpeak', 'slp', 'caa', 'thall'],
  },
];

type PredictState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'ok'; result: PredictionResponse }
  | { status: 'error'; message: string };

type FeatureForm = FormGroup<Record<string, FormControl<number | null>>>;

@Component({
  selector: 'app-prediction',
  imports: [ReactiveFormsModule, DecimalPipe, RiskGauge, RiskBadge, ErrorBanner, Skeleton],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './prediction.html',
})
export class Prediction {
  private readonly api = inject(ApiService);
  protected readonly status = inject(ApiStatusService);

  protected readonly features = loadable(() => this.api.features());
  protected readonly form = signal<FeatureForm | null>(null);
  protected readonly step = signal(0);
  protected readonly direction = signal<1 | -1>(1);
  protected readonly predict = signal<PredictState>({ status: 'idle' });

  protected readonly specs = computed(() => {
    const s = this.features.state();
    return s.status === 'ready' ? s.data : [];
  });

  protected readonly errorMessage = computed(() => {
    const s = this.features.state();
    return s.status === 'error' ? s.message : '';
  });

  protected readonly steps = computed<(FormStep & { specs: FeatureSpec[] })[]>(() => {
    const specs = this.specs();
    const byName = new Map(specs.map((f) => [f.name, f]));
    const known = new Set(STEPS.flatMap((s) => s.fields));
    const extra = specs.filter((f) => !known.has(f.name));
    return STEPS.map((s, i) => ({
      ...s,
      specs: [
        ...s.fields.map((n) => byName.get(n)).filter((f): f is FeatureSpec => !!f),
        ...(i === STEPS.length - 1 ? extra : []),
      ],
    }));
  });

  protected readonly current = computed(() => this.steps()[this.step()]);
  protected readonly isLast = computed(() => this.step() === STEPS.length - 1);
  /** Nota del modelo publicada por la API (metadata) o, si no hay, el texto por defecto. */
  protected readonly labelNote = computed(
    () => this.status.health()?.target_note || DEFAULT_TARGET_NOTE,
  );

  /** Umbral de decisión del modelo (de /api/metrics; opcional, no bloquea el formulario). */
  protected readonly threshold = signal<number | null>(null);

  protected readonly predictError = computed(() => {
    const p = this.predict();
    return p.status === 'error' ? p.message : '';
  });

  protected readonly result = computed(() => {
    const p = this.predict();
    return p.status === 'ok' ? p.result : null;
  });

  constructor() {
    this.api.metrics().subscribe({
      next: (m) => this.threshold.set(m.threshold ?? null),
      error: () => this.threshold.set(null),
    });
    effect(() => {
      const specs = this.specs();
      if (specs.length && !this.form()) this.form.set(buildForm(specs));
    });
  }

  control(name: string): FormControl<number | null> {
    return this.form()!.controls[name];
  }

  protected isInvalid(name: string): boolean {
    const c = this.form()?.controls[name];
    return !!c && c.invalid && (c.touched || c.dirty);
  }

  protected errorFor(spec: FeatureSpec): string {
    const c = this.form()?.controls[spec.name];
    if (!c?.errors) return '';
    if (c.errors['required'])
      return spec.type === 'categorical' ? 'Selecciona una opción.' : 'Este campo es obligatorio.';
    if (c.errors['min'] || c.errors['max'])
      return `Debe estar entre ${fmt(spec.min)} y ${fmt(spec.max)}.`;
    return 'Valor no válido.';
  }

  protected useSegmented(spec: FeatureSpec): boolean {
    const opts = spec.options ?? [];
    return opts.length <= 3 && opts.every((o) => o.label_es.length <= 18);
  }

  protected setOption(name: string, value: number): void {
    const c = this.control(name);
    c.setValue(value);
    c.markAsTouched();
  }

  protected next(): void {
    const form = this.form();
    if (!form) return;
    const fields = this.current().specs.map((s) => s.name);
    fields.forEach((f) => form.controls[f]?.markAsTouched());
    if (fields.some((f) => form.controls[f]?.invalid)) {
      focusFirstInvalid();
      return;
    }
    if (this.isLast()) {
      this.submit();
    } else {
      this.goTo(this.step() + 1);
    }
  }

  protected goTo(i: number): void {
    if (i === this.step() || i < 0 || i >= STEPS.length) return;
    this.direction.set(i > this.step() ? 1 : -1);
    this.step.set(i);
  }

  protected fillExample(): void {
    const form = this.form();
    if (!form) return;
    form.patchValue(EXAMPLE_PATIENT);
    form.markAllAsTouched();
  }

  protected reset(): void {
    const specs = this.specs();
    this.form.set(buildForm(specs));
    this.predict.set({ status: 'idle' });
    this.direction.set(-1);
    this.step.set(0);
  }

  submit(): void {
    const form = this.form();
    if (!form) return;
    form.markAllAsTouched();
    if (form.invalid) {
      const firstBad = this.steps().findIndex((s) =>
        s.specs.some((f) => form.controls[f.name].invalid),
      );
      if (firstBad >= 0) this.goTo(firstBad);
      return;
    }
    const payload = Object.fromEntries(
      Object.entries(form.getRawValue()).map(([k, v]) => [k, Number(v)]),
    ) as PatientInput;
    this.predict.set({ status: 'loading' });
    this.api.predict(payload).subscribe({
      next: (result) => this.predict.set({ status: 'ok', result }),
      error: (err: unknown) =>
        this.predict.set({ status: 'error', message: describeApiError(err) }),
    });
  }
}

const EXAMPLE_PATIENT: PatientInput = {
  age: 58,
  sex: 1,
  trtbps: 140,
  chol: 260,
  fbs: 0,
  cp: 0,
  exng: 1,
  restecg: 1,
  thalachh: 140,
  oldpeak: 1.6,
  slp: 1,
  caa: 1,
  thall: 3,
};

function fmt(v: number): string {
  return v.toLocaleString('es-ES');
}

function buildForm(specs: FeatureSpec[]): FeatureForm {
  const controls: Record<string, FormControl<number | null>> = {};
  for (const f of specs) {
    const initial = f.type === 'numeric' ? roundToStep((f.min + f.max) / 2, f.step, f.min) : null;
    controls[f.name] = new FormControl<number | null>(initial, [
      Validators.required,
      Validators.min(f.min),
      Validators.max(f.max),
    ]);
  }
  return new FormGroup(controls);
}

function roundToStep(v: number, step: number, min: number): number {
  const s = step || 1;
  const decimals = (String(s).split('.')[1] ?? '').length;
  return Number((Math.round((v - min) / s) * s + min).toFixed(decimals));
}

function focusFirstInvalid(): void {
  setTimeout(() => {
    const el = document.querySelector<HTMLElement>('[aria-invalid="true"]');
    el?.focus();
  });
}
