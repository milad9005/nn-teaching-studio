// Lesson 14 math: per-sample gradients, their mean (the batch gradient), and grouping the
// training history into epochs. All gradients come from the engine's backprop (batchLossGrad).

import { batchLossGrad, flattenGrads, predict, type Dataset, type LossName, type LossOptions, type Network } from '../../engine';

export interface SampleGrad {
  n: number;
  x: number[];
  y: number[];
  yhat: number[];
  loss: number;
  /** Flat gradient of this sample's loss, in engine parameter order. */
  g: number[];
}

export function perSampleGrads(net: Network, data: Dataset, loss: LossName, opts: LossOptions, idx: number[]): SampleGrad[] {
  return idx.map((n) => {
    const r = batchLossGrad(net, data, loss, opts, [n]);
    return { n, x: data.X[n], y: data.Y[n], yhat: predict(net, data.X[n]), loss: r.perSample[0], g: flattenGrads(net, r.grads) };
  });
}

/** Element-wise mean of the per-sample gradients. */
export function meanGrad(rows: { g: number[] }[]): number[] {
  if (!rows.length) return [];
  const out = new Array(rows[0].g.length).fill(0);
  rows.forEach((r) => r.g.forEach((v, k) => (out[k] += v / rows.length)));
  return out;
}

export const updatesPerEpoch = (N: number, B: number) => Math.ceil(N / (B <= 0 || B >= N ? N : B));

export interface EpochGroup {
  epoch: number;
  steps: { iter: number; batch: number[] }[];
}

/**
 * Groups history points by the epoch their batch belonged to. A point records the schedule
 * *after* its update, so the batch's epoch is the previous point's epoch.
 */
export function groupByEpoch(history: { iter: number; epoch: number; batch?: number[] }[]): EpochGroup[] {
  const out: EpochGroup[] = [];
  for (let i = 1; i < history.length; i++) {
    const h = history[i];
    if (!h.batch) continue;
    const epoch = history[i - 1].epoch;
    let g = out[out.length - 1];
    if (!g || g.epoch !== epoch) out.push((g = { epoch, steps: [] }));
    g.steps.push({ iter: h.iter, batch: h.batch });
  }
  return out;
}
