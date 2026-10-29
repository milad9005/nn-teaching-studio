import { describe, expect, it } from 'vitest';
import { loadPreset, sampleResult } from '../../engine';
import { finiteDiffs, lossAtW, outAtW, singleWeight, wDomain } from './l08';

describe('lesson 08: dL/dw for a single weight', () => {
  const p = loadPreset('single-weight');
  const x = p.data.X[0][0];
  const y = p.data.Y[0][0];

  it('preset numbers: ŷ = 6, L = 16, dŷ/dw = 2, dL/dw = −16', () => {
    const s = singleWeight(x, y, 3, 0);
    expect(s.yhat).toBe(6);
    expect(s.L).toBe(16);
    expect(s.dyhatdw).toBe(2);
    expect(s.dLdyhat).toBe(-8);
    expect(s.dLdw).toBe(-16);
    expect(s.wStar).toBe(5);
  });

  it('the closed form agrees with the engine (forward and backprop) for many w, x, y', () => {
    for (const [xx, yy, w] of [
      [2, 10, 3],
      [2, 10, 7],
      [-1.5, 4, 0.3],
      [0.5, -2, -4],
    ]) {
      const net = { ...p.net, layers: [{ ...p.net.layers[0], W: [[w]] }] };
      const r = sampleResult(net, [xx], [yy], 'mse');
      const s = singleWeight(xx, yy, w, 0);
      expect(r.trace.a[1][0]).toBeCloseTo(s.yhat, 12);
      expect(r.loss).toBeCloseTo(s.L, 12);
      expect(r.back.grads.W[0][0][0]).toBeCloseTo(s.dLdw, 12);
      expect(lossAtW(p.net, { X: [[xx]], Y: [[yy]] }, 'mse', w)).toBeCloseTo(s.L, 12);
      expect(outAtW(p.net, xx, w)).toBeCloseTo(s.yhat, 12);
    }
  });

  it('forward difference error is exactly x²·h; central difference is exact for this parabola', () => {
    const f = (w: number) => lossAtW(p.net, p.data, 'mse', w);
    for (const h of [1, 0.1, 0.01]) {
      const d = finiteDiffs(f, 3, h, -16);
      expect(d.errFwd).toBeCloseTo(x * x * h, 8);
      expect(Math.abs(d.errCen)).toBeLessThan(1e-8);
    }
  });

  it('the output slope is the same everywhere; the loss slope changes sign at w*', () => {
    expect(singleWeight(x, y, 3, 0).dyhatdw).toBe(singleWeight(x, y, 7, 0).dyhatdw);
    expect(singleWeight(x, y, 3, 0).dLdw).toBeLessThan(0);
    expect(singleWeight(x, y, 7, 0).dLdw).toBeGreaterThan(0);
    expect(singleWeight(x, y, 5, 0).dLdw).toBe(0);
  });

  it('x = 0 gives no target weight and a zero gradient', () => {
    const s = singleWeight(0, 10, 3, 0);
    expect(Number.isNaN(s.wStar)).toBe(true);
    expect(s.dLdw === 0).toBe(true);
  });

  it('w-domain always contains the default view and w*', () => {
    expect(wDomain(2, 10, 0)).toEqual([-2, 10]);
    const [lo, hi] = wDomain(0.5, 10, 0);
    expect(lo).toBe(-2);
    expect(hi).toBeGreaterThanOrEqual(23);
    expect(wDomain(0, 10, 0)).toEqual([-2, 10]);
  });
});
