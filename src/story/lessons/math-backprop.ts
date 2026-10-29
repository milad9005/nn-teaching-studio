// Lesson "backprop" math: the backward sweep on the two-neuron chain (the blame numbers in the order
// they are found: 4, 24, 8, 16), one update step, and the branching network where one knob reaches
// the output along two paths whose shares add up (3 + 3 = 6).

import { datasetLoss, pathsFrom, sampleResult, setParam, type Dataset, type LossName, type LossOptions, type Network } from '../../engine';
import { chainOf } from './math-chain';

export type BlameAt = 'out' | 'w2' | 'a1' | 'w1';

export interface Blame {
  at: BlameAt;
  /** The number found here. */
  value: number;
  /** The two numbers multiplied to get it: (blame that arrived) × (local rate). */
  from: number;
  rate: number;
}

/** The backward sweep on the chain, in the order the numbers are found. */
export function backwardSweep(net: Network, data: Dataset, loss: LossName = 'mse', lossOpts: LossOptions = {}): Blame[] {
  const c = chainOf(net, data, loss, lossOpts);
  return [
    // 2(ŷ − y): the loss's own rate; nothing arrives before it.
    { at: 'out', value: c.rLast, from: 1, rate: c.rLast },
    // w₂'s share: the blame at the output × what w₂ multiplies (a₁, through the ReLU).
    { at: 'w2', value: c.gw2, from: c.rLast, rate: c.a1 * c.relu2 },
    // Passed back to the first neuron: × w₂ (through the ReLU).
    { at: 'a1', value: c.dA1, from: c.rLast, rate: c.rMid },
    // w₁'s share: × x (through the first ReLU).
    { at: 'w1', value: c.gw1, from: c.dA1, rate: c.rFirst },
  ];
}

/** One gradient-descent step on both weights, w ← w − lr · dL/dw (biases left out of the picture). */
export function updateStep(net: Network, data: Dataset, lr: number, loss: LossName = 'mse', lossOpts: LossOptions = {}) {
  const c = chainOf(net, data, loss, lossOpts);
  return { w1: c.w1, w2: c.w2, g1: c.gw1, g2: c.gw2, w1New: c.w1 - lr * c.gw1, w2New: c.w2 - lr * c.gw2 };
}

/** The knob in the branching network that reaches the output along two paths: input → first hidden neuron. */
export const BRANCH_KNOB = { kind: 'W' as const, layer: 0, j: 0, i: 0, id: 'W1[0,0]' };

export interface Branching {
  x: number;
  y: number;
  /** The knob's value. */
  w: number;
  /** Hidden values: first layer (P, Q) and second layer (R, S). */
  h1: number[];
  h2: number[];
  yhat: number;
  L: number;
  /** ∂L/∂ŷ. */
  dOut: number;
  /** One entry per path from the knob to the output: the second-layer neuron it passes and its share. */
  paths: { via: number; share: number }[];
  sum: number;
  /** The engine's backprop answer for the knob. */
  grad: number;
}

export function branchingOf(net: Network, data: Dataset, loss: LossName = 'mse', lossOpts: LossOptions = {}): Branching {
  const one = sampleResult(net, data.X[0], data.Y[0], loss, lossOpts);
  const { layer, j, i } = BRANCH_KNOB;
  const paths = pathsFrom(net, one.trace, one.back.dz, layer, j, one.trace.a[layer][i]).map((p) => ({ via: p.steps[1].unit, share: p.contribution }));
  return {
    x: data.X[0][0],
    y: data.Y[0][0],
    w: net.layers[layer].W[j][i],
    h1: one.trace.a[1],
    h2: one.trace.a[2],
    yhat: one.trace.a[3][0],
    L: one.loss,
    dOut: one.dOut[0],
    paths,
    sum: paths.reduce((s, p) => s + p.share, 0),
    grad: one.back.grads.W[layer][j][i],
  };
}

/** Really turn the branching knob by h and run the network again. */
export function nudgeBranch(net: Network, data: Dataset, h: number, loss: LossName = 'mse', lossOpts: LossOptions = {}) {
  const w = net.layers[BRANCH_KNOB.layer].W[BRANCH_KNOB.j][BRANCH_KNOB.i];
  const before = datasetLoss(net, data, loss, lossOpts);
  const after = datasetLoss(setParam(net, BRANCH_KNOB, w + h), data, loss, lossOpts);
  return { from: w, to: w + h, before, after, change: after - before };
}
