import { TestBed } from '@angular/core/testing';

import { ConfusionMatrix, flareColor, inkFor } from './confusion-matrix';
import { RiskBadge } from './risk-badge';
import { RiskGauge } from './risk-gauge';

describe('RiskGauge', () => {
  it('expone la probabilidad en el aria-label', () => {
    const f = TestBed.createComponent(RiskGauge);
    f.componentRef.setInput('probability', 0.734);
    f.detectChanges();
    expect(f.nativeElement.querySelector('svg').getAttribute('aria-label')).toContain('73.4');
  });
});

describe('RiskBadge', () => {
  for (const [level, text] of [
    ['bajo', 'Riesgo bajo'],
    ['moderado', 'Riesgo moderado'],
    ['alto', 'Riesgo alto'],
  ] as const) {
    it(`nivel ${level}`, () => {
      const f = TestBed.createComponent(RiskBadge);
      f.componentRef.setInput('level', level);
      f.detectChanges();
      expect(f.nativeElement.textContent).toContain(text);
    });
  }
});

describe('ConfusionMatrix', () => {
  it('pinta las 4 celdas con porcentaje por fila', () => {
    const f = TestBed.createComponent(ConfusionMatrix);
    f.componentRef.setInput('matrix', [
      [15, 6],
      [3, 22],
    ]);
    f.detectChanges();
    const cells = f.nativeElement.querySelectorAll('[role="cell"]');
    expect(cells).toHaveLength(4);
    expect(cells[3].textContent).toContain('22');
    expect(cells[3].textContent).toContain('88%');
  });

  it('elige una tinta legible según el fondo', () => {
    expect(inkFor(flareColor(0))).toBe('#241027');
    expect(inkFor(flareColor(1))).toBe('#ffffff');
  });
});
