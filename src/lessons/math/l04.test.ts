import { describe, expect, it } from 'vitest';
import { datasetLoss, linspace, loadPreset, mulberry32, predict } from '../../engine';
import { breakpoint, buildNet, contribution, kinks, presetUnits, scenario, SCENARIOS, type Unit } from './l04';

const sum = (units: Unit[], c: number, x: number) => units.reduce((s, u) => s + contribution(u, x), c);

describe('l04: hidden ReLU layer as a sum of pieces', () => {
  it('engine predict equals c + sum of v_k ReLU(w_k x + b_k) for random networks', () => {
    const rng = mulberry32(42);
    const r = () => rng() * 6 - 3;
    for (let trial = 0; trial < 50; trial++) {
      const n = 1 + Math.floor(rng() * 10);
      const units = Array.from({ length: n }, () => ({ w: r(), b: r(), v: r() }));
      const c = r();
      const net = buildNet(units, c);
      for (const x of linspace(-3, 3, 13)) expect(predict(net, [x])[0]).toBeCloseTo(sum(units, c, x), 12);
    }
  });

  it('starts from the one-vs-many preset network', () => {
    const { units, c } = presetUnits();
    const preset = loadPreset('one-vs-many');
    for (const x of linspace(-2, 2, 9)) expect(predict(buildNet(units, c), [x])[0]).toBeCloseTo(predict(preset.net, [x])[0], 12);
  });

  it('breakpoints sit where w x + b = 0', () => {
    expect(breakpoint({ w: 2, b: -1, v: 1 })).toBeCloseTo(0.5);
    expect(breakpoint({ w: -1, b: -1.5, v: 1 })).toBeCloseTo(-1.5);
    expect(breakpoint({ w: 0, b: 1, v: 1 })).toBeNull();
  });

  it('counts only real bends: duplicates merge, cancelling pairs, zero v and off-screen hinges vanish', () => {
    expect(kinks([{ w: 1, b: 0, v: 1 }, { w: 2, b: 0, v: 0.5 }], -3, 3)).toEqual([{ x: 0, jump: 2 }]);
    expect(kinks([{ w: 1, b: 0, v: 1 }, { w: -1, b: 0, v: -1 }], -3, 3)).toEqual([]);
    expect(kinks([{ w: 1, b: -1, v: 0 }], -3, 3)).toEqual([]);
    expect(kinks([{ w: 1, b: -4, v: 1 }], -3, 3)).toEqual([]);
  });

  it('slope jump at a bend equals v |w|', () => {
    const units = [{ w: -2, b: 1, v: 0.7 }];
    const [k] = kinks(units, -3, 3);
    const h = 1e-6;
    const net = buildNet(units, 0);
    const f = (x: number) => predict(net, [x])[0];
    const jump = (f(k.x + 2 * h) - f(k.x + h)) / h - (f(k.x - h) - f(k.x - 2 * h)) / h;
    expect(jump).toBeCloseTo(k.jump, 5);
  });

  it('fit8 passes through all nine targets of 0.5 x^2 (MSE 0)', () => {
    const { units, n, c } = scenario('fit8');
    const data = loadPreset('one-vs-many').data;
    expect(datasetLoss(buildNet(units.slice(0, n), c), data, 'mse')).toBeLessThan(1e-20);
  });

  it('duplicate, off-screen and zero-v scenarios compute the same function as fit4 on screen', () => {
    const base = SCENARIOS.fit4();
    const f4 = buildNet(base.units, base.c);
    for (const id of ['duplicate', 'offscreen', 'zeroV'] as const) {
      const s = scenario(id);
      expect(s.n).toBeGreaterThan(base.units.length);
      const net = buildNet(s.units.slice(0, s.n), s.c);
      for (const x of linspace(-3, 3, 61)) expect(predict(net, [x])[0]).toBeCloseTo(predict(f4, [x])[0], 12);
      expect(kinks(s.units.slice(0, s.n), -3, 3).length).toBe(kinks(base.units, -3, 3).length);
    }
  });

  it('every scenario fills all 10 slots', () => {
    for (const id of Object.keys(SCENARIOS) as (keyof typeof SCENARIOS)[]) expect(scenario(id).units).toHaveLength(10);
  });
});
