import { describe, expect, it } from 'vitest';
import { datasetLoss } from '../../engine';
import { argMin, asFn, fitStory, noiseLoss, NEW_INPUTS, TRAIN_STEPS } from './math-generalize';

describe('lesson 12 · generalize', () => {
  const s = fitStory();

  it('the big network memorises the training dots: training loss almost 0', () => {
    expect(s.trainLoss.length).toBe(TRAIN_STEPS + 1);
    expect(s.trainLoss[TRAIN_STEPS]).toBeLessThan(0.001);
    expect(datasetLoss(s.endNet, s.data, 'mse')).toBeCloseTo(s.trainLoss[TRAIN_STEPS], 10);
  });

  it('but on the check data it is clearly worse than at the stopping point', () => {
    expect(s.stop).toBeGreaterThan(0);
    expect(s.stop).toBeLessThan(TRAIN_STEPS / 2);
    expect(s.valLoss[TRAIN_STEPS]).toBeGreaterThan(2 * s.valLoss[s.stop]);
    expect(s.valLoss[TRAIN_STEPS]).toBeGreaterThan(noiseLoss(s.val, s.truth));
  });

  it('the stopping network is the one with the lowest check loss', () => {
    expect(argMin([3, 1, 2])).toBe(1);
    expect(datasetLoss(s.stopNet, s.val, 'mse')).toBeCloseTo(Math.min(...s.valLoss), 10);
    expect(s.knobs).toBe(25);
  });

  it('new inputs are none of the training or check dots', () => {
    const seen = [...s.data.X, ...s.val.X].map((x) => x[0]);
    for (const x of NEW_INPUTS) {
      expect(seen.some((v) => Math.abs(v - x) < 1e-9)).toBe(false);
      expect(Number.isFinite(asFn(s.stopNet)(x))).toBe(true);
    }
  });
});

describe('lesson 12 · sizes', () => {
  it('preset sizes match the trained story', async () => {
    const { presetSizes } = await import('./math-generalize');
    const s = fitStory();
    expect(presetSizes()).toEqual({ knobs: s.knobs, dots: s.data.X.length, check: s.val.X.length });
  });
});

describe('lesson 12 · check dots', () => {
  it('are the same no matter how often the preset was loaded', async () => {
    const { checkDots } = await import('./math-generalize');
    const { loadPreset } = await import('../../engine');
    const a = checkDots();
    loadPreset('overfit');
    loadPreset('overfit');
    expect(checkDots()).toEqual(a);
    expect(a.X.length).toBe(14);
  });
});
