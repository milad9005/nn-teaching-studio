// Lesson 17 math: building networks of a chosen width, early stopping, and a complexity sweep.

import {
  datasetLoss,
  initialize,
  makeNetwork,
  newTrainState,
  trainStep,
  type Dataset,
  type Network,
  type TrainConfig,
} from '../../engine';

export const hiddenNet = (H: number, seed: number): Network => initialize(makeNetwork([1, H, 1], ['tanh', 'linear']), 'xavier', seed);

/** Index in the history with the lowest validation loss (-1 if there is none). */
export function bestValIndex(history: { val?: number }[]): number {
  let best = -1;
  history.forEach((h, i) => {
    if (h.val !== undefined && Number.isFinite(h.val) && (best < 0 || h.val < history[best].val!)) best = i;
  });
  return best;
}

export type Verdict = 'underfit' | 'good' | 'overfit';

/**
 * A rough reading of the two losses. The validation noise has std 0.25 (variance ≈ 0.06),
 * so no model can average much below that on unseen points.
 */
export function verdict(train: number, val: number): Verdict {
  if (train > 0.15) return 'underfit';
  if (val - train > 0.06) return 'overfit';
  return 'good';
}

export interface SweepRow {
  H: number;
  params: number;
  train: number;
  val: number;
  bestVal: number;
  bestIter: number;
}

/** Train one width for `steps` iterations with the given config; records the best validation loss. */
export function sweepOne(H: number, seed: number, steps: number, data: Dataset, val: Dataset, cfg: Omit<TrainConfig, 'mask'>): SweepRow {
  const net = hiddenNet(H, seed);
  let st = newTrainState(net, data, cfg);
  let bestVal = datasetLoss(net, val, cfg.loss, cfg.lossOpts);
  let bestIter = 0;
  let train = datasetLoss(net, data, cfg.loss, cfg.lossOpts);
  for (let i = 1; i <= steps; i++) {
    const r = trainStep(st, data, cfg);
    if (!r.ok) break;
    st = r.state;
    train = r.loss;
    const v = datasetLoss(st.net, val, cfg.loss, cfg.lossOpts);
    if (v < bestVal) {
      bestVal = v;
      bestIter = i;
    }
  }
  return { H, params: 3 * H + 1, train, val: datasetLoss(st.net, val, cfg.loss, cfg.lossOpts), bestVal, bestIter };
}
