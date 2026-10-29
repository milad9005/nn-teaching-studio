import { describe, expect, it } from 'vitest';
import { batchLossGrad, loadPreset } from '../../engine';
import { angleBetween, bestLine, dataStory, lossAt, lossContour, meanOf } from './math-data';

describe('lesson 10 · data', () => {
  const s = dataStory();

  it('the average of the per-example steps is the step for all examples together', () => {
    const p = loadPreset('mini-batch');
    const g = batchLossGrad(p.net, p.data, p.loss, p.lossOpts).grads;
    expect(s.mean[0]).toBeCloseTo(-s.lr * g.W[0][0][0], 10);
    expect(s.mean[1]).toBeCloseTo(-s.lr * g.b[0][0], 10);
    expect(meanOf([[1, 2], [3, 6]])).toEqual([2, 4]);
  });

  it('a batch points roughly, not exactly, the same way as all the examples', () => {
    const a = angleBetween(s.mean, s.batchMean);
    expect(a).toBeGreaterThan(1);
    expect(a).toBeLessThan(30);
  });

  it('epoch 1 uses every example exactly once, in 4 batches of 2', () => {
    expect(s.perEpoch).toBe(4);
    expect(s.epoch1.flat().sort()).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(s.frames[1].batch).toEqual(s.epoch1[0]);
    expect(s.frames[4].batch).toEqual(s.epoch1[3]);
  });

  it('training settles the line onto the best line through the data', () => {
    const last = s.frames[s.frames.length - 1];
    expect(s.frames.length).toBe(12 * 4 + 1);
    expect(last.loss).toBeLessThan(s.frames[0].loss / 100);
    expect(Math.abs(last.w - s.best.w)).toBeLessThan(0.1);
    expect(Math.abs(last.b - s.best.b)).toBeLessThan(0.1);
    expect(bestLine({ X: [[0], [1], [2]], Y: [[1], [3], [5]] })).toEqual({ w: 2, b: 1 });
  });

  it('the batch loss is bumpy: it sometimes goes up', () => {
    const ups = s.frames.slice(2).filter((f, i) => f.batchLoss > s.frames[i + 1].batchLoss).length;
    expect(ups).toBeGreaterThan(3);
  });
});

describe('lesson 10 · loss contours', () => {
  it('every contour point has exactly the contour loss', () => {
    const s = dataStory();
    for (const level of [0.5, 2, 6]) for (const [w, b] of lossContour(s.data, level, 12)) expect(lossAt(s.data, w, b)).toBeCloseTo(level, 9);
    expect(lossContour(s.data, 0)).toEqual([]);
  });
});
