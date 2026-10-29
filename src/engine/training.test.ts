import { describe, expect, it } from 'vitest';
import {
  PRESET_IDS,
  datasetLoss,
  defaultOptimizer,
  flatten,
  initOptState,
  initialize,
  loadPreset,
  makeNetwork,
  maskFor,
  newTrainState,
  optimizePath,
  optimizerStep,
  sampleResult,
  trainStep,
  validate,
  advance,
  newSchedule,
  nextBatch,
  type Preset,
  type TrainConfig,
} from './index';

const close = (a: number, b: number, tol = 1e-9) => expect(Math.abs(a - b)).toBeLessThan(tol);
const cfgOf = (p: Preset): TrainConfig => ({
  loss: p.loss,
  lossOpts: p.lossOpts,
  opt: p.opt,
  batch: p.batch,
  mask: maskFor(p.net, p.frozen),
  divergeAt: 1e8,
});

describe('optimizers', () => {
  it('SGD uses the selected learning rate', () => {
    const r = optimizerStep([3], [-16], defaultOptimizer('sgd', 0.05), initOptState(1));
    expect(r.ok && r.theta[0]).toBeCloseTo(3.8, 12);
  });
  it('momentum accumulates velocity', () => {
    const cfg = { ...defaultOptimizer('momentum', 0.1), momentum: 0.5 };
    const a = optimizerStep([0], [1], cfg, initOptState(1));
    if (!a.ok) throw new Error();
    close(a.theta[0], -0.1);
    const b = optimizerStep(a.theta, [1], cfg, a.state);
    if (!b.ok) throw new Error();
    close(b.theta[0], -0.1 - 0.1 * 1.5);
  });
  it('Adam first step moves by ~lr regardless of gradient scale', () => {
    for (const g of [1e-3, 1, 1e3]) {
      const r = optimizerStep([0], [g], defaultOptimizer('adam', 0.01), initOptState(1));
      if (!r.ok) throw new Error();
      close(r.theta[0], -0.01, 1e-7);
    }
  });
  it('mask freezes parameters and clipping caps the norm', () => {
    const cfg = { ...defaultOptimizer('sgd', 1), clipNorm: 1 };
    const r = optimizerStep([0, 0], [3, 4], cfg, initOptState(2), [true, false]);
    if (!r.ok) throw new Error();
    expect(r.theta[1]).toBe(0);
    close(r.theta[0], -1); // [3,0] clipped to norm 1
  });
  it('rejects non-finite gradients', () => {
    expect(optimizerStep([0], [NaN], defaultOptimizer(), initOptState(1)).ok).toBe(false);
  });
  it('learning-rate regimes on L(w) = (2w - 10)^2', () => {
    const f = (p: number[]) => (2 * p[0] - 10) ** 2;
    const g = (p: number[]) => [4 * (2 * p[0] - 10)];
    const run = (lr: number) => optimizePath(f, g, [3], defaultOptimizer('sgd', lr), 40);
    expect(run(0.01).losses.at(-1)!).toBeGreaterThan(0.01); // slow: still far from 0 after 40 steps
    expect(run(0.1).losses.at(-1)!).toBeLessThan(1e-6); // fast
    const osc = run(0.2).points.map((p) => p[0] - 5);
    expect(Math.sign(osc[1])).toBe(-Math.sign(osc[2])); // overshoots back and forth
    expect(run(0.26).diverged || run(0.26).losses.at(-1)! > 16).toBe(true); // diverges
  });
});

describe('batch schedule', () => {
  it('visits every sample once per epoch', () => {
    const cfg = { size: 3, shuffle: true, seed: 2 };
    let s = newSchedule(8, cfg);
    const seen: number[] = [];
    while (s.epoch === 0) {
      seen.push(...nextBatch(8, cfg, s));
      s = advance(8, cfg, s);
    }
    expect(seen.sort()).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(s.iteration).toBe(3);
  });
});

describe('presets', () => {
  it('all presets are valid', () => {
    for (const id of PRESET_IDS) {
      const p = loadPreset(id);
      expect(validate(p.net, p.data), id).toEqual([]);
      if (p.val) expect(validate(p.net, p.val)).toEqual([]);
    }
  });

  it('presets are deterministic', () => {
    for (const id of PRESET_IDS) expect(flatten(loadPreset(id).net)).toEqual(flatten(loadPreset(id).net));
  });

  it('single-affine: z = 2·1.5 + 1 = 4, loss 1', () => {
    const p = loadPreset('single-affine');
    const r = sampleResult(p.net, p.data.X[0], p.data.Y[0], p.loss);
    expect(r.trace.a[1][0]).toBe(4);
    expect(r.loss).toBe(1);
  });

  it('relu-threshold outputs', () => {
    const p = loadPreset('relu-threshold');
    expect(p.data.X.map((x) => sampleResult(p.net, x, [0], 'mse').trace.a[1][0])).toEqual([0, 0, 0, 1]);
  });

  it('one-vs-many: output at x = 2 is 1.5', () => {
    const p = loadPreset('one-vs-many');
    expect(sampleResult(p.net, [2], [2], 'mse').trace.a[2][0]).toBe(1.5);
  });

  it('temperature: MSE 0.045', () => {
    const p = loadPreset('temperature');
    close(datasetLoss(p.net, p.data, 'mse'), 0.045, 1e-12);
  });

  it('two-weight: loss 37/6 and gradient (-10/3, -11/3)', () => {
    const p = loadPreset('two-weight');
    close(datasetLoss(p.net, p.data, 'mse'), 37 / 6);
    const st = newTrainState(p.net, p.data, cfgOf(p));
    const r = trainStep(st, p.data, cfgOf(p));
    if (!r.ok) throw new Error();
    close(r.grads.W[0][0][0], -10 / 3);
    close(r.grads.W[0][0][1], -11 / 3);
    expect(r.state.net.layers[0].b[0]).toBe(0); // frozen
  });

  it('weight-bias: loss 16 and gradient (dL/dw, dL/db) = (-16, -8)', () => {
    const p = loadPreset('weight-bias');
    close(datasetLoss(p.net, p.data, 'mse'), 16);
    const r = trainStep(newTrainState(p.net, p.data, cfgOf(p)), p.data, cfgOf(p));
    if (!r.ok) throw new Error();
    close(r.grads.W[0][0][0], -16);
    close(r.grads.b[0][0], -8);
  });

  it('gd-step: w 3 -> 3.8, loss 16 -> 5.76', () => {
    const p = loadPreset('gd-step');
    const r = trainStep(newTrainState(p.net, p.data, cfgOf(p)), p.data, cfgOf(p));
    if (!r.ok) throw new Error();
    expect(r.batchLoss).toBe(16);
    close(r.state.net.layers[0].W[0][0], 3.8);
    close(r.loss, 5.76);
  });

  it('chain-two trains toward the target', () => {
    const p = loadPreset('chain-two');
    let st = newTrainState(p.net, p.data, cfgOf(p));
    for (let i = 0; i < 50; i++) {
      const r = trainStep(st, p.data, cfgOf(p));
      if (!r.ok) throw new Error(r.error);
      st = r.state;
    }
    expect(datasetLoss(st.net, p.data, 'mse')).toBeLessThan(1e-3);
  });

  it('mini-batch: one epoch is 4 updates of 2 samples', () => {
    const p = loadPreset('mini-batch');
    let st = newTrainState(p.net, p.data, cfgOf(p));
    const batches: number[][] = [];
    for (let i = 0; i < 4; i++) {
      const r = trainStep(st, p.data, cfgOf(p));
      if (!r.ok) throw new Error();
      batches.push(r.batch);
      st = r.state;
    }
    expect(batches.every((b) => b.length === 2)).toBe(true);
    expect(batches.flat().sort()).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(st.schedule.epoch).toBe(1);
  });

  it('overfit: a wide net drives training loss well below validation loss', () => {
    const p = loadPreset('overfit');
    const wide = initialize(makeNetwork([1, 10, 1], ['tanh', 'linear']), 'xavier', 4);
    let st = newTrainState(wide, p.data, cfgOf(p));
    for (let i = 0; i < 3000; i++) {
      const r = trainStep(st, p.data, cfgOf(p));
      if (!r.ok) throw new Error();
      st = r.state;
    }
    const train = datasetLoss(st.net, p.data, 'mse');
    const val = datasetLoss(st.net, p.val!, 'mse');
    expect(train).toBeLessThan(0.02);
    expect(val).toBeGreaterThan(train * 3);
  });

  it('divergence is reported, never silently produces NaN', () => {
    const p = loadPreset('lr-compare');
    const cfg = { ...cfgOf(p), opt: defaultOptimizer('sgd', 5), divergeAt: 1e6 };
    let st = newTrainState(p.net, p.data, cfg);
    let error = '';
    for (let i = 0; i < 50 && !error; i++) {
      const r = trainStep(st, p.data, cfg);
      if (r.ok) st = r.state;
      else error = r.error;
    }
    expect(error).toBe('diverged');
  });
});
