import { describe, expect, it } from 'vitest';
import { loadPreset } from '../../engine';
import { chainOf, nudgeChain } from './math-chain';

describe('lesson chain: the two-neuron chain', () => {
  const p = loadPreset('chain-two');
  const c = chainOf(p.net, p.data, p.loss, p.lossOpts);

  it('forward: 2 → 6 → 12, target 10, loss 4', () => {
    expect([c.x, c.a1, c.yhat, c.y, c.L]).toEqual([2, 6, 12, 10, 4]);
  });

  it('the three rates 4, 2, 2 multiply to dL/dw₁ = 16, as backprop says', () => {
    expect([c.rLast, c.rMid, c.rFirst]).toEqual([4, 2, 2]);
    expect(c.rLast * c.rMid * c.rFirst).toBe(16);
    expect(c.gw1).toBeCloseTo(16, 12);
  });

  it('dL/dw₂ = 4 · 6 = 24', () => {
    expect(c.rLast * c.a1).toBe(24);
    expect(c.gw2).toBeCloseTo(24, 12);
  });

  it('a real nudge agrees: w₁ + 0.001 → loss + 0.016', () => {
    const n = nudgeChain(p.net, p.data, 'w1', 0.001, p.loss, p.lossOpts);
    expect(n.change).toBeCloseTo(0.016, 4);
  });
});
