// Lesson 13 math: the same one-weight problem trained with several learning rates.
// Every trajectory comes from real engine training steps (trainStep), never from the closed form;
// the closed form is only used by the tests to verify them.

import {
  batchLossGrad,
  datasetLoss,
  defaultOptimizer,
  flattenGrads,
  maskFor,
  newTrainState,
  paramRefs,
  setParam,
  trainStep,
  type Dataset,
  type LossName,
  type LossOptions,
  type Network,
  type TrainConfig,
} from '../../engine';

export interface LrProblem {
  net: Network;
  data: Dataset;
  loss: LossName;
  lossOpts: LossOptions;
  frozen: string[];
}

export interface LrRun {
  lr: number;
  /** w_k for k = 0..(steps done). */
  w: number[];
  /** Full-dataset loss at w_k. */
  loss: number[];
  /** dL/dw at w_k (the gradient used for step k -> k+1). */
  grad: number[];
  /** True when training stopped because the loss exploded. */
  diverged: boolean;
}

/** The trained weight: W1[0,0] (the lesson's network has exactly one trainable parameter). */
const W_REF = { kind: 'W' as const, layer: 0, j: 0, i: 0, id: 'W1[0,0]' };
const wIndex = (net: Network) => paramRefs(net).findIndex((p) => p.id === W_REF.id);

export function runLr(p: LrProblem, lr: number, steps: number, divergeAt = 1e9): LrRun {
  const cfg: TrainConfig = {
    loss: p.loss,
    lossOpts: p.lossOpts,
    opt: defaultOptimizer('sgd', lr),
    batch: { size: 0, shuffle: false, seed: 1 },
    mask: maskFor(p.net, p.frozen),
    divergeAt,
  };
  const k = wIndex(p.net);
  let st = newTrainState(p.net, p.data, cfg);
  const w = [p.net.layers[0].W[0][0]];
  const loss = [datasetLoss(p.net, p.data, p.loss, p.lossOpts)];
  const grad: number[] = [];
  let diverged = false;
  for (let i = 0; i < steps; i++) {
    const r = trainStep(st, p.data, cfg);
    if (!r.ok) {
      diverged = true;
      break;
    }
    grad.push(flattenGrads(st.net, r.grads)[k]);
    st = r.state;
    w.push(st.net.layers[0].W[0][0]);
    loss.push(r.loss);
  }
  // Gradient at the last visited point too, so every w_k has its slope.
  const last = setParam(p.net, W_REF, w[w.length - 1]);
  grad.push(flattenGrads(last, batchLossGrad(last, p.data, p.loss, p.lossOpts).grads)[k]);
  return { lr, w, loss, grad, diverged };
}

/** L(w) for the lesson's network with every other parameter as it is. */
export const lossAt = (p: LrProblem, w: number) => datasetLoss(setParam(p.net, W_REF, w), p.data, p.loss, p.lossOpts);

/** dL/dw from backprop at a given w. */
export const gradAt = (p: LrProblem, w: number) => {
  const n = setParam(p.net, W_REF, w);
  return flattenGrads(n, batchLossGrad(n, p.data, p.loss, p.lossOpts).grads)[wIndex(n)];
};

/** Curvature L''(w), measured as a central difference of the backprop gradient. */
export const curvatureAt = (p: LrProblem, w: number, h = 1e-4) => (gradAt(p, w + h) - gradAt(p, w - h)) / (2 * h);

/** Where L'(w) = 0, assuming a quadratic (one Newton step from w, exact for this lesson). */
export const minimizerFrom = (p: LrProblem, w: number) => w - gradAt(p, w) / curvatureAt(p, w);

export type Regime = 'none' | 'slow' | 'fast' | 'oneStep' | 'oscillate' | 'edge' | 'diverge';

/**
 * On a quadratic with curvature c, one step maps (w - w*) to r·(w - w*) with r = 1 - lr·c.
 * |r| < 1 converges, r < 0 flips sides, r = 0 lands in one step, |r| > 1 diverges.
 */
export const factor = (lr: number, curvature: number) => 1 - lr * curvature;

export function regime(lr: number, curvature: number, tol = 1e-6): Regime {
  if (!(lr > 0) || !(curvature > 0)) return 'none';
  const r = factor(lr, curvature);
  if (Math.abs(r) < 0.02) return 'oneStep';
  if (Math.abs(r + 1) < tol) return 'edge';
  if (r < -1) return 'diverge';
  if (r < 0) return 'oscillate';
  return r >= 0.75 ? 'slow' : 'fast';
}

/** First step k at which the run's loss is below `tol`, or -1. */
export const stepsTo = (run: LrRun, tol: number) => run.loss.findIndex((l) => l < tol);
