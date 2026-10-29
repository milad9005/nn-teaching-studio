import { describe, expect, it } from 'vitest';
import { batchLossGrad, gradCheck, loadPreset, maxRelErr, paramRefs, pathsFrom, sampleResult, sizes, sumContrib } from '../../engine';
import { backwardMacs, forwardMacs, gradientCost, paramCount, pathCount, phaseOf } from './l11';

describe('lesson 11: backprop on the branching network', () => {
  const p = loadPreset('branching');
  const r = sampleResult(p.net, p.data.X[0], p.data.Y[0], 'mse');

  it('verified numbers: output 2.5, L = 2.25, dz2 = [3, 6], dW1 = [[6], [−3]]', () => {
    expect(r.trace.a[3][0]).toBe(2.5);
    expect(r.loss).toBe(2.25);
    expect(r.dOut[0]).toBe(3);
    expect(r.back.dz[1]).toEqual([3, 6]);
    expect(r.back.grads.W[0]).toEqual([[6], [-3]]);
    expect(r.back.grads.b[0]).toEqual([6, -3]);
  });

  it('W1[0,0] has two paths of 3 each; their sum is the gradient', () => {
    const paths = pathsFrom(p.net, r.trace, r.back.dz, 0, 0, r.trace.a[0][0]);
    expect(paths).toHaveLength(2);
    expect(paths.map((q) => q.contribution)).toEqual([3, 3]);
    expect(sumContrib(paths)).toBe(r.back.grads.W[0][0][0]);
    expect(pathCount(sizes(p.net), 1)).toBe(paths.length);
  });

  it('finite differences agree with backprop', () => {
    expect(maxRelErr(gradCheck(p.net, p.data, 'mse'))).toBeLessThan(1e-6);
  });

  it('cost model: 13 parameters, 8 forward MACs', () => {
    const s = sizes(p.net);
    expect(paramCount(s)).toBe(paramRefs(p.net).length);
    expect(forwardMacs(s)).toBe(1 * 2 + 2 * 2 + 2 * 1);
    expect(backwardMacs(s)).toBe(8 + (8 - 2));
    const c = gradientCost(s);
    expect(c.fdCentral).toBe(2 * 13 * 8);
    expect(c.backprop).toBe(8 + 14);
    // The gap grows with size: backprop is ~3 passes, finite differences ~2P passes.
    const big = gradientCost([784, 128, 64, 10]);
    expect(big.fdCentral / big.backprop).toBeGreaterThan(30000);
  });

  it('phase of the stepper', () => {
    expect(phaseOf(null, 3)).toBe('idle');
    expect(phaseOf({ phase: 'forward', k: 1 }, 3)).toBe('forward');
    expect(phaseOf({ phase: 'forward', k: 4 }, 3)).toBe('forward');
    expect(phaseOf({ phase: 'forward', k: 5 }, 3)).toBe('loss');
    expect(phaseOf({ phase: 'backward', k: 2 }, 3)).toBe('backward');
    expect(phaseOf({ phase: 'update', k: 0 }, 3)).toBe('update');
  });

  it('batch gradient equals single-sample gradient for one example', () => {
    expect(batchLossGrad(p.net, p.data, 'mse').grads.W[0]).toEqual(r.back.grads.W[0]);
  });
});
