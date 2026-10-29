import { describe, expect, it } from 'vitest';
import {
  batchLossGrad,
  defaultOptimizer,
  flattenGrads,
  loadPreset,
  maskFor,
  newTrainState,
  optimizePath,
  overfitTruth,
  trainStep,
} from '../../engine';
import { curvatureAt, factor, lossAt, minimizerFrom, regime, runLr, stepsTo, type LrProblem } from './l13';
import { groupByEpoch, meanGrad, perSampleGrads, updatesPerEpoch } from './l14';
import { contourGrid, makeObjective, tracePath } from './l15';
import { SYM_DATA, deepStats, heStd, symNet, trainFrom, unitSpread, xavierR } from './l16';
import { bestValIndex, hiddenNet, sweepOne, verdict } from './l17';
import { costOf, iterationCost } from './l18';

type HistoryLike = { iter: number; epoch: number; batch?: number[] };

const lrProblem = (): LrProblem => {
  const p = loadPreset('lr-compare');
  return { net: p.net, data: p.data, loss: p.loss, lossOpts: p.lossOpts, frozen: p.frozen };
};

describe('L13 learning rate', () => {
  const p = lrProblem();
  it('L(w) = (2w - 10)^2 with curvature 8 and minimum at w = 5', () => {
    expect(lossAt(p, 3)).toBeCloseTo(16, 10);
    expect(curvatureAt(p, 3)).toBeCloseTo(8, 6);
    expect(minimizerFrom(p, 3)).toBeCloseTo(5, 6);
  });
  it('engine trajectories match w_k = 5 + (w0 - 5)(1 - 8 lr)^k', () => {
    for (const lr of [0.01, 0.1, 0.125, 0.2, 0.24]) {
      const run = runLr(p, lr, 25);
      expect(run.diverged).toBe(false);
      run.w.forEach((w, k) => expect(w).toBeCloseTo(5 + (3 - 5) * (1 - 8 * lr) ** k, 8));
      run.grad.forEach((g, k) => expect(g).toBeCloseTo(8 * (run.w[k] - 5), 6));
    }
  });
  it('regimes: 0.01 slow, 0.1 fast, 0.125 one step, 0.2 oscillates, 0.25 bounces, 0.3 diverges', () => {
    expect(regime(0.01, 8)).toBe('slow');
    expect(regime(0.1, 8)).toBe('fast');
    expect(regime(0.125, 8)).toBe('oneStep');
    expect(regime(0.2, 8)).toBe('oscillate');
    expect(regime(0.25, 8)).toBe('edge');
    expect(regime(0.3, 8)).toBe('diverge');
    expect(regime(0, 8)).toBe('none');
    expect(factor(0.2, 8)).toBeCloseTo(-0.6, 12);
  });
  it('one step at lr = 1/8; slow run still far after 30 steps; large lr diverges', () => {
    expect(runLr(p, 0.125, 1).loss[1]).toBeLessThan(1e-20);
    expect(stepsTo(runLr(p, 0.125, 5), 1e-6)).toBe(1);
    expect(runLr(p, 0.01, 30).loss[30]).toBeGreaterThan(0.1);
    const osc = runLr(p, 0.2, 4).w.map((w) => Math.sign(w - 5));
    expect(osc.slice(1)).toEqual([1, -1, 1, -1]);
    const big = runLr(p, 0.3, 200);
    expect(big.diverged).toBe(true);
    expect(big.loss[big.loss.length - 1]).toBeGreaterThan(1e6);
  });
});

describe('L14 batches', () => {
  const p = loadPreset('mini-batch');
  it('the batch gradient is the mean of the per-sample gradients', () => {
    const idx = [3, 6];
    const rows = perSampleGrads(p.net, p.data, p.loss, p.lossOpts, idx);
    const engine = flattenGrads(p.net, batchLossGrad(p.net, p.data, p.loss, p.lossOpts, idx).grads);
    meanGrad(rows).forEach((g, k) => expect(g).toBeCloseTo(engine[k], 12));
    // Per-sample MSE gradient of a linear unit: dL/dw = 2(ŷ - y)x, dL/db = 2(ŷ - y).
    rows.forEach((r) => {
      expect(r.g[0]).toBeCloseTo(2 * (r.yhat[0] - r.y[0]) * r.x[0], 12);
      expect(r.g[1]).toBeCloseTo(2 * (r.yhat[0] - r.y[0]), 12);
    });
  });
  it('updates per epoch', () => {
    expect(updatesPerEpoch(8, 2)).toBe(4);
    expect(updatesPerEpoch(8, 3)).toBe(3);
    expect(updatesPerEpoch(8, 0)).toBe(1);
    expect(updatesPerEpoch(8, 1)).toBe(8);
  });
  it('groups real training history into epochs that each visit every sample once', () => {
    for (const size of [2, 3]) {
      const cfg = { loss: p.loss, lossOpts: p.lossOpts, opt: p.opt, batch: { ...p.batch, size }, mask: maskFor(p.net, []), divergeAt: 1e8 };
      let st = newTrainState(p.net, p.data, cfg);
      const history: HistoryLike[] = [{ iter: 0, epoch: 0 }];
      for (let i = 0; i < 3 * updatesPerEpoch(8, size); i++) {
        const r = trainStep(st, p.data, cfg);
        if (!r.ok) throw new Error(r.error);
        st = r.state;
        history.push({ iter: st.schedule.iteration, epoch: st.schedule.epoch, batch: r.batch });
      }
      const groups = groupByEpoch(history);
      expect(groups.map((g) => g.epoch)).toEqual([0, 1, 2]);
      groups.forEach((g) => {
        expect(g.steps.length).toBe(updatesPerEpoch(8, size));
        expect(g.steps.flatMap((s) => s.batch).sort()).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
      });
    }
  });
});

describe('L15 optimizers', () => {
  const valley = makeObjective('valley', { a: 0.4, b: 10, c: 10 });
  const curved = makeObjective('curved', { a: 0.4, b: 10, c: 10 });
  it('traces match the engine optimizePath', () => {
    for (const name of ['sgd', 'momentum', 'adam'] as const) {
      const cfg = defaultOptimizer(name, 0.05);
      const a = tracePath(valley, cfg, 30);
      const b = optimizePath(valley.f, valley.grad, valley.start, cfg, 30);
      a.points.forEach((pt, k) => pt.forEach((v, i) => expect(v).toBeCloseTo(b.points[k][i], 12)));
    }
  });
  it('gradients are correct (central differences)', () => {
    for (const obj of [valley, curved]) {
      const p = [0.3, -0.7];
      const g = obj.grad(p);
      [0, 1].forEach((i) => {
        const e = [0, 0];
        e[i] = 1e-6;
        const num = (obj.f([p[0] + e[0], p[1] + e[1]]) - obj.f([p[0] - e[0], p[1] - e[1]])) / 2e-6;
        expect(g[i]).toBeCloseTo(num, 5);
      });
    }
  });
  it('Adam: the first step moves every coordinate by ≈ lr against the gradient sign', () => {
    const t = tracePath(valley, defaultOptimizer('adam', 0.3), 1);
    expect(t.points[1][0] - t.points[0][0]).toBeCloseTo(0.3, 6);
    expect(t.points[1][1] - t.points[0][1]).toBeCloseTo(-0.3, 6);
  });
  it('momentum keeps v = βv + g', () => {
    const t = tracePath(valley, { ...defaultOptimizer('momentum', 0.04), momentum: 0.9 }, 3);
    expect(t.states[2].v[0]).toBeCloseTo(0.9 * t.grads[0][0] + t.grads[1][0], 12);
  });
  it('SGD is capped by the steep direction: lr > 2/b diverges', () => {
    expect(tracePath(valley, defaultOptimizer('sgd', 0.21), 200).diverged).toBe(true);
    expect(tracePath(valley, defaultOptimizer('sgd', 0.18), 200).diverged).toBe(false);
  });
  it('on the curved valley, momentum and Adam reach much lower loss than SGD in 300 steps', () => {
    const sgd = tracePath(curved, defaultOptimizer('sgd', 0.005), 300).losses.at(-1)!;
    const mom = tracePath(curved, { ...defaultOptimizer('momentum', 0.002), momentum: 0.9 }, 300).losses.at(-1)!;
    const adam = tracePath(curved, defaultOptimizer('adam', 0.05), 300).losses.at(-1)!;
    expect(mom).toBeLessThan(sgd / 10);
    expect(adam).toBeLessThan(sgd / 10);
  });
  it('contour grid has the requested shape', () => {
    const g = contourGrid(valley, 20);
    expect(g.z.length).toBe(20);
    expect(g.z[0].length).toBe(20);
  });
});

describe('L16 initialization', () => {
  it('zero init: hidden weights never move and the loss is stuck', () => {
    const r = trainFrom(symNet('zeros', 3, 1), SYM_DATA, 200);
    r.units.forEach((u) => u.forEach((q) => (expect(q.w).toBe(0), expect(q.v).toBe(0), expect(q.b).toBe(0))));
    expect(r.loss[200]).toBeCloseTo(r.loss[1], 10);
  });
  it('constant init: units move but stay identical copies forever', () => {
    const r = trainFrom(symNet('constant', 3, 0.5), SYM_DATA, 200);
    expect(r.units[200][0].w).not.toBeCloseTo(0.5, 3);
    r.units.forEach((u) => expect(unitSpread(u)).toBeLessThan(1e-12));
  });
  it('random inits break symmetry and fit much better', () => {
    const c = trainFrom(symNet('constant', 3, 0.5), SYM_DATA, 300).loss.at(-1)!;
    for (const init of ['uniform', 'xavier', 'he'] as const) {
      const r = trainFrom(symNet(init, 3, 1), SYM_DATA, 300);
      expect(unitSpread(r.units[300])).toBeGreaterThan(0.1);
      expect(r.loss.at(-1)!).toBeLessThan(c / 5);
    }
  });
  it('depth: small init vanishes, large saturates tanh, xavier and he keep a usable spread', () => {
    const small = deepStats('tanh', 'small');
    expect(small[5].std).toBeLessThan(0.01);
    const large = deepStats('tanh', 'large');
    expect(large[5].extreme).toBeGreaterThan(0.3);
    expect(deepStats('tanh', 'xavier')[5].std).toBeGreaterThan(0.15);
    const he = deepStats('relu', 'he');
    he.forEach((s) => (expect(s.std).toBeGreaterThan(0.3), expect(s.std).toBeLessThan(3)));
    expect(deepStats('relu', 'large')[5].std).toBeGreaterThan(5);
    expect(he[0].hist.counts.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
  });
  it('xavier and he scales', () => {
    expect(xavierR(1, 4)).toBeCloseTo(Math.sqrt(6 / 5), 12);
    expect(heStd(16)).toBeCloseTo(Math.sqrt(2 / 16), 12);
  });
});

describe('L17 overfitting', () => {
  const p = loadPreset('overfit');
  const cfg = { loss: p.loss, lossOpts: p.lossOpts, opt: p.opt, batch: p.batch, divergeAt: 1e8 };
  it('best validation index', () => {
    expect(bestValIndex([{ val: 3 }, { val: 1 }, { val: 2 }, {}])).toBe(1);
    expect(bestValIndex([{}, {}])).toBe(-1);
  });
  it('one unit underfits, ten units overfit, early stopping beats the final weights', () => {
    const one = sweepOne(1, 4, 1500, p.data, p.val!, cfg);
    expect(verdict(one.train, one.val)).toBe('underfit');
    const ten = sweepOne(10, 4, 1500, p.data, p.val!, cfg);
    expect(ten.train).toBeLessThan(0.01);
    expect(verdict(ten.train, ten.val)).toBe('overfit');
    expect(ten.bestVal).toBeLessThan(ten.val - 0.03);
    expect(hiddenNet(3, 4).layers[0].W.length).toBe(3);
  });
  it('validation targets follow the true function', () => {
    const resid = p.val!.X.map((x, i) => p.val!.Y[i][0] - overfitTruth(x[0]));
    expect(Math.max(...resid.map(Math.abs))).toBeLessThan(1);
  });
});

describe('L18 cost', () => {
  it('counts multiply-adds per layer', () => {
    expect(costOf([1, 1])).toEqual({ params: 2, forward: 1, backward: 1, update: 2 });
    const c = costOf([1, 4, 1]);
    expect(c.params).toBe(13);
    expect(c.forward).toBe(8);
    expect(c.backward).toBe(4 + 8);
    expect(iterationCost([1, 4, 1], 4)).toEqual({ training: 4 * (8 + 12) + 13, inference: 8, cost: c });
  });
});
