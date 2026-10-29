// Lesson 08 math: one weight, one example. The network is ŷ = w·x + b (b frozen), the loss is
// L = (ŷ − y)². All curves are evaluated through the engine so they match backprop exactly.

import { centralDiff, datasetLoss, forwardDiff, predict, setParam, type Dataset, type LossName, type Network, type ParamRef } from '../../engine';

export const W_REF: ParamRef = { kind: 'W', layer: 0, j: 0, i: 0, id: 'W1[0,0]' };

/** Loss of the whole (one-example) dataset with the weight replaced by `w`. */
export const lossAtW = (net: Network, data: Dataset, loss: LossName, w: number) => datasetLoss(setParam(net, W_REF, w), data, loss);

/** Network output for input x with the weight replaced by `w`. */
export const outAtW = (net: Network, x: number, w: number) => predict(setParam(net, W_REF, w), [x])[0];

/** The closed-form quantities of the lesson, written exactly as they appear on screen. */
export function singleWeight(x: number, y: number, w: number, b: number) {
  const yhat = w * x + b;
  const r = yhat - y;
  const dLdyhat = 2 * r;
  return {
    yhat,
    L: r * r,
    dLdyhat,
    dyhatdw: x,
    dLdw: dLdyhat * x,
    /** The weight that makes ŷ = y (undefined when x = 0: w has no effect). */
    wStar: Math.abs(x) > 1e-9 ? (y - b) / x : NaN,
  };
}

/** Forward and central finite differences of f at w with step h, and their errors against `exact`. */
export function finiteDiffs(f: (w: number) => number, w: number, h: number, exact: number) {
  const fwd = forwardDiff(f, w, h);
  const cen = centralDiff(f, w, h);
  return { fwd, cen, errFwd: fwd - exact, errCen: cen - exact };
}

/**
 * A plotting range for w that contains the default view [-2, 10] and the target weight w*.
 * Depends only on x, y, b (not on w), so the axes stay still while the learner drags.
 */
export function wDomain(x: number, y: number, b: number): [number, number] {
  const ws = Math.abs(x) > 0.15 ? (y - b) / x : NaN;
  if (!Number.isFinite(ws)) return [-2, 10];
  return [Math.max(-60, Math.min(-2, ws - 3)), Math.min(60, Math.max(10, ws + 3))];
}
