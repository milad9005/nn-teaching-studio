import { describe, expect, it } from 'vitest';
import { loadPreset, trainStep, newTrainState, maskFor, datasetLoss } from '../../engine';
import { backwardSweep, branchingOf, nudgeBranch, updateStep } from './math-backprop';

describe('lesson backprop', () => {
  const p = loadPreset('chain-two');

  it('the backward sweep finds 4, 24, 8, 16 in that order', () => {
    const s = backwardSweep(p.net, p.data, p.loss, p.lossOpts);
    expect(s.map((b) => b.at)).toEqual(['out', 'w2', 'a1', 'w1']);
    s.forEach((b, i) => expect(b.value).toBeCloseTo([4, 24, 8, 16][i], 12));
    s.forEach((b) => expect(b.from * b.rate).toBeCloseTo(b.value, 12));
    expect(s.map((b) => [b.from, b.rate]).slice(1)).toEqual([
      [4, 6],
      [4, 2],
      [8, 2],
    ]);
  });

  it('one update step moves the knobs against their slopes', () => {
    const u = updateStep(p.net, p.data, p.opt.lr, p.loss, p.lossOpts);
    expect(u.w1New).toBeCloseTo(3 - 0.01 * 16, 12);
    expect(u.w2New).toBeCloseTo(2 - 0.01 * 24, 12);
  });

  it('training the chain drives the loss towards 0', () => {
    const cfg = { loss: p.loss, lossOpts: p.lossOpts, opt: p.opt, batch: p.batch, mask: maskFor(p.net, p.frozen), divergeAt: 1e8 };
    let st = newTrainState(p.net, p.data, cfg);
    for (let i = 0; i < 60; i++) {
      const r = trainStep(st, p.data, cfg);
      if (!r.ok) throw new Error('training failed');
      st = r.state;
    }
    expect(datasetLoss(st.net, p.data, p.loss, p.lossOpts)).toBeLessThan(0.01);
  });

  it('branching: two paths of 3 and 3 add up to 6, and a real nudge agrees', () => {
    const b = loadPreset('branching');
    const r = branchingOf(b.net, b.data, b.loss, b.lossOpts);
    expect(r.paths.map((q) => q.share)).toEqual([3, 3]);
    expect(r.sum).toBe(6);
    expect(r.grad).toBeCloseTo(6, 12);
    expect(nudgeBranch(b.net, b.data, 0.001, b.loss, b.lossOpts).change).toBeCloseTo(0.006, 4);
  });
});
