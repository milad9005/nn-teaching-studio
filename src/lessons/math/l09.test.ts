import { describe, expect, it } from 'vitest';
import { batchLossGrad, linspace, loadPreset } from '../../engine';
import { B_REF, contourLevels, contourSegments, lossAt, lossGrid, withWeights } from './l09';
import { setParam } from '../../engine';

describe('lesson 09: two-weight loss surface', () => {
  const p = loadPreset('two-weight');

  it('start: L = 37/6, gradient (−10/3, −11/3); minimum L = 0 at (2, 1)', () => {
    expect(lossAt(p.net, p.data, 'mse', 0.5, -1)).toBeCloseTo(37 / 6, 12);
    expect(lossAt(p.net, p.data, 'mse', 2, 1)).toBe(0);
    const g = batchLossGrad(p.net, p.data, 'mse').grads.W[0][0];
    expect(g[0]).toBeCloseTo(-10 / 3, 12);
    expect(g[1]).toBeCloseTo(-11 / 3, 12);
  });

  it('closed form L = ((w1−2)² + (w2−1)² + (w1+w2−3)²)/3 matches the engine', () => {
    for (const [a, b] of [
      [0, 0],
      [3, -1.5],
      [-0.7, 2.2],
    ])
      expect(lossAt(p.net, p.data, 'mse', a, b)).toBeCloseTo(((a - 2) ** 2 + (b - 1) ** 2 + (a + b - 3) ** 2) / 3, 12);
  });

  it('grid minimum sits at (2, 1)', () => {
    const w1s = linspace(-1, 5, 41);
    const w2s = linspace(-2, 4, 41);
    const z = lossGrid(p.net, p.data, 'mse', w1s, w2s);
    let best = { v: Infinity, i: 0, j: 0 };
    z.forEach((row, j) => row.forEach((v, i) => v < best.v && (best = { v, i, j })));
    expect(w1s[best.i]).toBeCloseTo(2, 12);
    expect(w2s[best.j]).toBeCloseTo(1, 12);
  });

  it('a slice at another bias has its lowest point elsewhere and above 0', () => {
    const nb = setParam(p.net, B_REF, 1);
    const w1s = linspace(-1, 5, 61);
    const w2s = linspace(-2, 4, 61);
    const z = lossGrid(nb, p.data, 'mse', w1s, w2s);
    const min = Math.min(...z.flat());
    expect(min).toBeGreaterThan(0.01);
    expect(lossAt(withWeights(nb, 2, 1), p.data, 'mse', 2, 1)).toBeCloseTo(1, 12);
  });

  it('marching squares traces a circle for x² + y² = 1', () => {
    const xs = linspace(-2, 2, 81);
    const ys = linspace(-2, 2, 81);
    const z = ys.map((y) => xs.map((x) => x * x + y * y));
    const segs = contourSegments(xs, ys, z, 1);
    expect(segs.length).toBeGreaterThan(40);
    for (const s of segs) for (const [x, y] of s) expect(Math.abs(Math.hypot(x, y) - 1)).toBeLessThan(0.01);
  });

  it('contour levels are increasing and below the max', () => {
    const lv = contourLevels(20, 8);
    expect(lv).toHaveLength(8);
    lv.forEach((v, k) => k && expect(v).toBeGreaterThan(lv[k - 1]));
    expect(lv[7]).toBeLessThan(20);
    expect(contourLevels(0)).toEqual([]);
  });
});
