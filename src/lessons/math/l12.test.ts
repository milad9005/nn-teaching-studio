import { describe, expect, it } from 'vitest';
import { batchLossGrad, defaultOptimizer, loadPreset, maskFor, newTrainState, optimizePath, trainStep, type TrainConfig } from '../../engine';
import { closedFormW, curvature, gdFactor, sgdUpdate, wStar } from './l12';

describe('lesson 12: gradient descent on (2w − 10)²', () => {
  const p = loadPreset('gd-step');
  const cfg = (lr: number): TrainConfig => ({ loss: p.loss, lossOpts: p.lossOpts, opt: { ...p.opt, lr }, batch: p.batch, mask: maskFor(p.net, p.frozen), divergeAt: 1e8 });

  it('verified first step: dL/dw = −16, w 3 → 3.8, loss 16 → 5.76', () => {
    const g = batchLossGrad(p.net, p.data, 'mse').grads.W[0][0][0];
    expect(g).toBe(-16);
    expect(sgdUpdate(3, 0.05, g)).toBeCloseTo(3.8, 12);
    const r = trainStep(newTrainState(p.net, p.data, cfg(0.05)), p.data, cfg(0.05));
    if (!r.ok) throw new Error(r.error);
    expect(r.state.net.layers[0].W[0][0]).toBeCloseTo(3.8, 12);
    expect(r.loss).toBeCloseTo(5.76, 12);
    expect(r.state.net.layers[0].b[0]).toBe(0);
  });

  it('closed form matches the engine for several learning rates', () => {
    for (const lr of [0.01, 0.05, 0.1, 0.2]) {
      let st = newTrainState(p.net, p.data, cfg(lr));
      for (let k = 1; k <= 8; k++) {
        const r = trainStep(st, p.data, cfg(lr));
        if (!r.ok) throw new Error(r.error);
        st = r.state;
        expect(st.net.layers[0].W[0][0]).toBeCloseTo(closedFormW(3, k, lr, 2, 10, 0), 9);
      }
    }
  });

  it('curvature 8, factor 1 − 8η; η = 1/8 lands on w* = 5 in one step', () => {
    expect(curvature(2)).toBe(8);
    expect(gdFactor(0.05, 2)).toBeCloseTo(0.6, 12);
    expect(wStar(2, 10, 0)).toBe(5);
    expect(closedFormW(3, 1, 0.125, 2, 10, 0)).toBe(5);
  });

  it('η above 2/L″ = 0.25 diverges', () => {
    const f = (q: number[]) => (2 * q[0] - 10) ** 2;
    const g = (q: number[]) => [4 * (2 * q[0] - 10)];
    expect(Math.abs(gdFactor(0.3, 2))).toBeGreaterThan(1);
    const run = optimizePath(f, g, [3], defaultOptimizer('sgd', 0.3), 200, 1e6);
    expect(run.diverged).toBe(true);
  });
});
