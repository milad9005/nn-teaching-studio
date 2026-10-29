// Tools for explaining gradients: finite-difference checks and chain-rule path decomposition.

import { actDeriv, isKink } from './activations';
import {
  batchLossGrad,
  flattenGrads,
  forward,
  getParam,
  paramRefs,
  setParam,
  type Dataset,
  type Network,
  type ParamRef,
  type Trace,
} from './network';
import type { LossName, LossOptions } from './losses';

/** Central difference of a scalar function. */
export const centralDiff = (f: (x: number) => number, x: number, h = 1e-5) => (f(x + h) - f(x - h)) / (2 * h);
export const forwardDiff = (f: (x: number) => number, x: number, h: number) => (f(x + h) - f(x)) / h;

export const relErr = (a: number, b: number) => Math.abs(a - b) / Math.max(1e-8, Math.abs(a) + Math.abs(b));

export interface GradCheckRow {
  param: ParamRef;
  analytic: number;
  numeric: number;
  relErr: number;
  /** True when a ReLU input sits within h of 0, so the finite difference crosses a kink. */
  nearKink: boolean;
}

export function gradCheck(
  net: Network,
  data: Dataset,
  loss: LossName,
  opts: LossOptions = {},
  idx?: number[],
  h = 1e-5,
): GradCheckRow[] {
  const { grads } = batchLossGrad(net, data, loss, opts, idx);
  const refs = paramRefs(net);
  const analytic = flattenGrads(net, grads);
  const ids = idx ?? data.X.map((_, i) => i);
  return refs.map((p, k) => {
    const v = getParam(net, p);
    const f = (t: number) => batchLossGrad(setParam(net, p, t), data, loss, opts, idx).loss;
    const numeric = centralDiff(f, v, h);
    // A kink is crossed if some ReLU input changes sign (or touches 0) between v-h and v+h.
    const lo = setParam(net, p, v - h);
    const hi = setParam(net, p, v + h);
    const nearKink = ids.some((s) => {
      const zl = forward(lo, data.X[s]).z;
      const zh = forward(hi, data.X[s]).z;
      return net.layers.some(
        (layer, l) =>
          layer.act === 'relu' &&
          zl[l].some((z, j) => isKink('relu', z) || isKink('relu', zh[l][j]) || Math.sign(z) !== Math.sign(zh[l][j])),
      );
    });
    return { param: p, analytic: analytic[k], numeric, relErr: relErr(analytic[k], numeric), nearKink };
  });
}

export const maxRelErr = (rows: GradCheckRow[]) =>
  rows.filter((r) => !r.nearKink).reduce((m, r) => Math.max(m, Math.abs(r.analytic - r.numeric) < 1e-9 ? 0 : r.relErr), 0);

// ---------------------------------------------------------------------------
// Chain-rule paths.

export interface PathStep {
  /** Layer of the unit this step enters (0-based layer index). */
  layer: number;
  unit: number;
  /** Weight crossed to reach this unit (undefined for the starting unit). */
  weight?: { layer: number; j: number; i: number; value: number };
  /** f'(z) of this unit (for the last unit this is folded into dL/dz). */
  actDeriv: number;
}

export interface GradPath {
  steps: PathStep[];
  /** dL/dz of the final output unit on this path. */
  dLdzOut: number;
  /** Product of all local derivatives on the path, times dL/dz_out (and times the input for a weight). */
  contribution: number;
  /** The factors in order, for display: [input a_i (weights only), f'(z) , w, f'(z), w, ..., dL/dz_out]. */
  factors: number[];
}

/**
 * All paths from unit (l, j) to the output units, used to show where a gradient comes from.
 * The sum of the contributions equals dL/dz[l][j] (or dL/dW when `inputActivation` is given).
 * Output dL/dz already includes the output activation (and softmax, if used).
 */
export function pathsFrom(
  net: Network,
  tr: Trace,
  dz: number[][],
  l: number,
  j: number,
  inputActivation?: number,
  maxPaths = 64,
): GradPath[] {
  const L = net.layers.length;
  const out: GradPath[] = [];
  const walk = (layer: number, unit: number, steps: PathStep[], prod: number, factors: number[]) => {
    if (out.length >= maxPaths) return;
    if (layer === L - 1) {
      const dOut = dz[layer][unit];
      const last = steps[steps.length - 1];
      last.actDeriv = 1;
      out.push({ steps, dLdzOut: dOut, contribution: prod * dOut, factors: [...factors, dOut] });
      return;
    }
    const d = actDeriv(net.layers[layer].act, tr.z[layer][unit]);
    steps[steps.length - 1].actDeriv = d;
    const nextLayer = net.layers[layer + 1];
    for (let k = 0; k < nextLayer.b.length; k++) {
      const w = nextLayer.W[k][unit];
      walk(
        layer + 1,
        k,
        [...steps, { layer: layer + 1, unit: k, weight: { layer: layer + 1, j: k, i: unit, value: w }, actDeriv: 1 }],
        prod * d * w,
        [...factors, d, w],
      );
    }
  };
  const start = inputActivation ?? 1;
  walk(l, j, [{ layer: l, unit: j, actDeriv: 1 }], start, inputActivation === undefined ? [] : [inputActivation]);
  return out;
}

export const sumContrib = (paths: GradPath[]) => paths.reduce((s, p) => s + p.contribution, 0);

/** Numerically differentiate any scalar function of the flat parameter vector. */
export function numericGradFlat(net: Network, f: (n: Network) => number, h = 1e-5): number[] {
  const refs = paramRefs(net);
  return refs.map((p) => {
    const v = getParam(net, p);
    return (f(setParam(net, p, v + h)) - f(setParam(net, p, v - h))) / (2 * h);
  });
}

