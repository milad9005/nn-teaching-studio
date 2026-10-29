import { describe, expect, it } from 'vitest';
import { batchLossGrad, datasetLoss, loadPreset, lossSample, mulberry32 } from '../../engine';
import { fitLine, lineNet, OUTLIER, withOutlier, type RegLoss } from './l06';

const temp = loadPreset('temperature').data;

describe('l06: best lines for regression losses', () => {
  it('least squares on the temperature data matches the hand calculation (w = 1.36, b = -2.19)', () => {
    const f = fitLine('mse', temp, 1);
    expect(f.w).toBeCloseTo(1.36, 10);
    expect(f.b).toBeCloseTo(-2.19, 10);
  });

  it('the MSE and Huber fits have zero gradient (engine batchLossGrad)', () => {
    for (const on of [false, true]) {
      const data = withOutlier(temp, on);
      for (const [loss, delta] of [['mse', 1], ['huber', 0.3], ['huber', 1]] as [RegLoss, number][]) {
        const f = fitLine(loss, data, delta);
        const g = batchLossGrad(lineNet(f.w, f.b), data, loss, { delta }).grads;
        expect(Math.abs(g.W[0][0][0])).toBeLessThan(1e-9);
        expect(Math.abs(g.b[0][0])).toBeLessThan(1e-9);
      }
    }
  });

  it('no nearby line beats any fitted line (MSE, MAE, Huber)', () => {
    const rng = mulberry32(7);
    for (const on of [false, true])
      for (const loss of ['mse', 'mae', 'huber'] as RegLoss[]) {
        const data = withOutlier(temp, on);
        const f = fitLine(loss, data, 0.5);
        for (let t = 0; t < 400; t++) {
          const dw = (rng() - 0.5) * 0.4;
          const db = (rng() - 0.5) * 0.4;
          expect(datasetLoss(lineNet(f.w + dw, f.b + db), data, loss, { delta: 0.5 })).toBeGreaterThanOrEqual(f.loss - 1e-12);
        }
      }
  });

  it('with a large delta Huber equals half the squared error, so its fit equals least squares', () => {
    const data = withOutlier(temp, true);
    const a = fitLine('huber', data, 100);
    const b = fitLine('mse', data, 1);
    expect(a.w).toBeCloseTo(b.w, 9);
    expect(a.b).toBeCloseTo(b.b, 9);
    expect(lossSample('huber', [2], [0.5], { delta: 100 })).toBeCloseTo(0.5 * lossSample('mse', [2], [0.5]), 12);
  });

  it('the outlier pulls the MSE line much more than the MAE line', () => {
    const clean = { mse: fitLine('mse', temp, 1), mae: fitLine('mae', temp, 1) };
    const dirty = { mse: fitLine('mse', withOutlier(temp, true), 1), mae: fitLine('mae', withOutlier(temp, true), 1) };
    const at = (p: { w: number; b: number }) => p.w * OUTLIER.x + p.b;
    expect(Math.abs(at(dirty.mse) - at(clean.mse))).toBeGreaterThan(3 * Math.abs(at(dirty.mae) - at(clean.mae)));
  });
});
