// Lesson 06 math: the line y = w x + b that minimizes each regression loss on a small dataset,
// so the lesson can show "the best you could do" next to the learner's hand-tuned line.
// Loss values themselves always come from the engine (lossSample / batchLossGrad / datasetLoss).

import { datasetLoss, type Dataset, type Network } from '../../engine';

export type RegLoss = 'mse' | 'mae' | 'huber';

/** A broken-sensor reading added to the temperature data to show robustness. */
export const OUTLIER = { x: 2.25, y: 3.2 };

export const withOutlier = (data: Dataset, on: boolean): Dataset =>
  on ? { X: [...data.X, [OUTLIER.x]], Y: [...data.Y, [OUTLIER.y]] } : data;

/** The single linear neuron yhat = w x + b as an engine network. */
export const lineNet = (w: number, b: number): Network => ({ inputSize: 1, layers: [{ W: [[w]], b: [b], act: 'linear' }] });

/** Weighted least squares for a line. Falls back to a flat line when all x coincide. */
function wls(xs: number[], ys: number[], ws: number[]): { w: number; b: number } {
  const S = ws.reduce((a, v) => a + v, 0);
  const mx = xs.reduce((a, x, i) => a + ws[i] * x, 0) / S;
  const my = ys.reduce((a, y, i) => a + ws[i] * y, 0) / S;
  let sxx = 0;
  let sxy = 0;
  xs.forEach((x, i) => {
    sxx += ws[i] * (x - mx) ** 2;
    sxy += ws[i] * (x - mx) * (ys[i] - my);
  });
  const w = sxx > 1e-15 ? sxy / sxx : 0;
  return { w, b: my - w * mx };
}

/** Closed-form least squares: the minimizer of MSE. */
export const fitMSE = (xs: number[], ys: number[]) => wls(xs, ys, xs.map(() => 1));

/**
 * Least absolute deviations. MAE is convex and piecewise linear in (w, b), so some minimizer is a
 * vertex: a line through two data points with different x. Trying every pair is exact.
 */
export function fitMAE(xs: number[], ys: number[]): { w: number; b: number } {
  const mae = (w: number, b: number) => xs.reduce((a, x, i) => a + Math.abs(w * x + b - ys[i]), 0) / xs.length;
  let best = fitMSE(xs, ys);
  let bestV = mae(best.w, best.b);
  for (let i = 0; i < xs.length; i++)
    for (let j = i + 1; j < xs.length; j++) {
      if (Math.abs(xs[i] - xs[j]) < 1e-12) continue;
      const w = (ys[j] - ys[i]) / (xs[j] - xs[i]);
      const b = ys[i] - w * xs[i];
      const v = mae(w, b);
      if (v < bestV - 1e-15) {
        best = { w, b };
        bestV = v;
      }
    }
  return best;
}

/** Huber regression by iteratively reweighted least squares (weight 1 inside delta, delta/|r| outside). */
export function fitHuber(xs: number[], ys: number[], delta: number): { w: number; b: number } {
  let p = fitMSE(xs, ys);
  for (let it = 0; it < 500; it++) {
    const ws = xs.map((x, i) => {
      const r = Math.abs(p.w * x + p.b - ys[i]);
      return r <= delta ? 1 : delta / r;
    });
    const q = wls(xs, ys, ws);
    const done = Math.abs(q.w - p.w) + Math.abs(q.b - p.b) < 1e-13;
    p = q;
    if (done) break;
  }
  return p;
}

export function fitLine(loss: RegLoss, data: Dataset, delta: number): { w: number; b: number; loss: number } {
  const xs = data.X.map((x) => x[0]);
  const ys = data.Y.map((y) => y[0]);
  const p = loss === 'mse' ? fitMSE(xs, ys) : loss === 'mae' ? fitMAE(xs, ys) : fitHuber(xs, ys, delta);
  return { ...p, loss: datasetLoss(lineNet(p.w, p.b), data, loss, { delta }) };
}
