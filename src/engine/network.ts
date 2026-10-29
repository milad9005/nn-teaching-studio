// Small dense feed-forward networks: forward pass, backward pass, parameter access.
//
// Shapes. A network with inputSize n0 and layers of sizes n1..nL has, for layer l (0-based):
//   W[l]: n(l+1) rows x n(l) cols   (W[l][j][i] connects unit i of the previous layer to unit j)
//   b[l]: n(l+1)
// Forward trace for one sample x:  a[0] = x,  z[l] = W[l] a[l] + b[l],  a[l+1] = f_l(z[l]).

import { activateLayer, backLayer, type ActivationName } from './activations';
import { lossGrad, lossSample, type LossName, type LossOptions } from './losses';

export interface Layer {
  W: number[][];
  b: number[];
  act: ActivationName;
}

export interface Network {
  inputSize: number;
  layers: Layer[];
}

export interface Dataset {
  X: number[][];
  Y: number[][];
}

export interface Trace {
  /** a[0] is the input; a[l+1] is the output of layer l. */
  a: number[][];
  z: number[][];
}

export interface Grads {
  W: number[][][];
  b: number[][];
}

export interface Backward {
  grads: Grads;
  /** dL/dz for every layer. */
  dz: number[][];
  /** dL/da; da[0] is dL/dx and da[L] is dL/d(output). */
  da: number[][];
}

export const LIMITS = { maxInputs: 4, maxHiddenLayers: 4, maxUnits: 10, maxOutputs: 4 };

export const sizes = (net: Network) => [net.inputSize, ...net.layers.map((l) => l.b.length)];

export const cloneNet = (net: Network): Network => ({
  inputSize: net.inputSize,
  layers: net.layers.map((l) => ({ W: l.W.map((r) => r.slice()), b: l.b.slice(), act: l.act })),
});

export function forward(net: Network, x: number[]): Trace {
  const a: number[][] = [x.slice()];
  const z: number[][] = [];
  net.layers.forEach((layer, l) => {
    const prev = a[l];
    const zl = layer.W.map((row, j) => row.reduce((s, w, i) => s + w * prev[i], layer.b[j]));
    z.push(zl);
    a.push(activateLayer(layer.act, zl));
  });
  return { a, z };
}

export const predict = (net: Network, x: number[]) => {
  const t = forward(net, x);
  return t.a[t.a.length - 1];
};

/**
 * Backward pass for one sample, starting from dL/d(output).
 * `scale` multiplies the incoming gradient (1/N when averaging over a batch).
 */
export function backward(net: Network, tr: Trace, dOut: number[], scale = 1): Backward {
  const L = net.layers.length;
  const da: number[][] = new Array(L + 1);
  const dz: number[][] = new Array(L);
  const W: number[][][] = new Array(L);
  const b: number[][] = new Array(L);
  da[L] = dOut.map((v) => v * scale);
  for (let l = L - 1; l >= 0; l--) {
    const layer = net.layers[l];
    dz[l] = backLayer(layer.act, tr.z[l], tr.a[l + 1], da[l + 1]);
    W[l] = dz[l].map((d) => tr.a[l].map((ai) => d * ai));
    b[l] = dz[l].slice();
    da[l] = tr.a[l].map((_, i) => layer.W.reduce((s, row, j) => s + row[i] * dz[l][j], 0));
  }
  return { grads: { W, b }, dz, da };
}

export interface SampleResult {
  trace: Trace;
  loss: number;
  dOut: number[];
  back: Backward;
}

export function sampleResult(
  net: Network,
  x: number[],
  y: number[],
  loss: LossName,
  opts: LossOptions = {},
): SampleResult {
  const trace = forward(net, x);
  const out = trace.a[trace.a.length - 1];
  const dOut = lossGrad(loss, out, y, opts);
  return { trace, loss: lossSample(loss, out, y, opts), dOut, back: backward(net, trace, dOut) };
}

export const zeroGrads = (net: Network): Grads => ({
  W: net.layers.map((l) => l.W.map((r) => r.map(() => 0))),
  b: net.layers.map((l) => l.b.map(() => 0)),
});

export function addGrads(acc: Grads, g: Grads, k = 1) {
  acc.W.forEach((Wl, l) => Wl.forEach((r, j) => r.forEach((_, i) => (r[i] += k * g.W[l][j][i]))));
  acc.b.forEach((bl, l) => bl.forEach((_, j) => (bl[j] += k * g.b[l][j])));
}

/** Mean loss and mean gradient over the listed samples (all samples if idx is omitted). */
export function batchLossGrad(
  net: Network,
  data: Dataset,
  loss: LossName,
  opts: LossOptions = {},
  idx?: number[],
): { loss: number; grads: Grads; perSample: number[] } {
  const ids = idx ?? data.X.map((_, i) => i);
  const N = ids.length;
  const grads = zeroGrads(net);
  const perSample: number[] = [];
  let total = 0;
  for (const n of ids) {
    const tr = forward(net, data.X[n]);
    const out = tr.a[tr.a.length - 1];
    const l = lossSample(loss, out, data.Y[n], opts);
    perSample.push(l);
    total += l;
    addGrads(grads, backward(net, tr, lossGrad(loss, out, data.Y[n], opts), 1 / N).grads);
  }
  return { loss: total / N, grads, perSample };
}

export function datasetLoss(net: Network, data: Dataset, loss: LossName, opts: LossOptions = {}): number {
  if (!data.X.length) return NaN;
  let s = 0;
  data.X.forEach((x, n) => (s += lossSample(loss, predict(net, x), data.Y[n], opts)));
  return s / data.X.length;
}

// ---------------------------------------------------------------------------
// Parameters as a flat vector (used by optimizers and gradient checks).

export interface ParamRef {
  kind: 'W' | 'b';
  layer: number;
  /** Unit in this layer. */
  j: number;
  /** Unit in the previous layer (weights only). */
  i: number;
  /** Stable id like "W1[0,1]" or "b2[0]" with 1-based layer numbers. */
  id: string;
}

export function paramRefs(net: Network): ParamRef[] {
  const out: ParamRef[] = [];
  net.layers.forEach((layer, l) => {
    layer.W.forEach((row, j) =>
      row.forEach((_, i) => out.push({ kind: 'W', layer: l, j, i, id: `W${l + 1}[${j},${i}]` })),
    );
    layer.b.forEach((_, j) => out.push({ kind: 'b', layer: l, j, i: -1, id: `b${l + 1}[${j}]` }));
  });
  return out;
}

export const getParam = (net: Network, p: ParamRef) =>
  p.kind === 'W' ? net.layers[p.layer].W[p.j][p.i] : net.layers[p.layer].b[p.j];

export const getGrad = (g: Grads, p: ParamRef) => (p.kind === 'W' ? g.W[p.layer][p.j][p.i] : g.b[p.layer][p.j]);

export function setParam(net: Network, p: ParamRef, v: number): Network {
  const n = cloneNet(net);
  if (p.kind === 'W') n.layers[p.layer].W[p.j][p.i] = v;
  else n.layers[p.layer].b[p.j] = v;
  return n;
}

export const flatten = (net: Network) => paramRefs(net).map((p) => getParam(net, p));
export const flattenGrads = (net: Network, g: Grads) => paramRefs(net).map((p) => getGrad(g, p));

export function unflatten(net: Network, v: number[]): Network {
  const n = cloneNet(net);
  paramRefs(n).forEach((p, k) => {
    if (p.kind === 'W') n.layers[p.layer].W[p.j][p.i] = v[k];
    else n.layers[p.layer].b[p.j] = v[k];
  });
  return n;
}

// ---------------------------------------------------------------------------
// Validation.

export type ConfigError =
  | { code: 'inputDim'; expected: number; got: number; sample: number }
  | { code: 'targetDim'; expected: number; got: number; sample: number }
  | { code: 'shape'; layer: number }
  | { code: 'softmaxHidden'; layer: number }
  | { code: 'nonFinite' }
  | { code: 'tooLarge' }
  | { code: 'emptyData' };

export function validate(net: Network, data?: Dataset): ConfigError[] {
  const errs: ConfigError[] = [];
  const s = sizes(net);
  net.layers.forEach((layer, l) => {
    if (layer.W.length !== s[l + 1] || layer.W.some((r) => r.length !== s[l])) errs.push({ code: 'shape', layer: l });
    if (layer.act === 'softmax' && l !== net.layers.length - 1) errs.push({ code: 'softmaxHidden', layer: l });
  });
  if (
    net.inputSize > LIMITS.maxInputs ||
    net.layers.length - 1 > LIMITS.maxHiddenLayers ||
    s.slice(1).some((n) => n > LIMITS.maxUnits || n < 1)
  )
    errs.push({ code: 'tooLarge' });
  if (flatten(net).some((v) => !Number.isFinite(v))) errs.push({ code: 'nonFinite' });
  if (data) {
    if (!data.X.length) errs.push({ code: 'emptyData' });
    const out = s[s.length - 1];
    data.X.forEach((x, n) => {
      if (x.length !== net.inputSize) errs.push({ code: 'inputDim', expected: net.inputSize, got: x.length, sample: n });
      if (data.Y[n]?.length !== out) errs.push({ code: 'targetDim', expected: out, got: data.Y[n]?.length ?? 0, sample: n });
    });
  }
  return errs;
}

// ---------------------------------------------------------------------------
// Building and editing architectures.

export function makeNetwork(
  layerSizes: number[],
  acts: ActivationName[],
  fill: (l: number, j: number, i: number) => number = () => 0,
  biasFill: (l: number, j: number) => number = () => 0,
): Network {
  return {
    inputSize: layerSizes[0],
    layers: layerSizes.slice(1).map((n, l) => ({
      W: Array.from({ length: n }, (_, j) => Array.from({ length: layerSizes[l] }, (_, i) => fill(l, j, i))),
      b: Array.from({ length: n }, (_, j) => biasFill(l, j)),
      act: acts[l],
    })),
  };
}

/** Insert a hidden unit in layer l (not the output layer). New weights start at `w0`. */
export function addUnit(net: Network, l: number, w0 = 0.1): Network {
  const n = cloneNet(net);
  const layer = n.layers[l];
  layer.W.push(new Array(layer.W[0]?.length ?? sizes(net)[l]).fill(w0));
  layer.b.push(0);
  n.layers[l + 1].W.forEach((row) => row.push(w0));
  return n;
}

export function removeUnit(net: Network, l: number, j: number): Network {
  const n = cloneNet(net);
  if (n.layers[l].b.length <= 1) return n;
  n.layers[l].W.splice(j, 1);
  n.layers[l].b.splice(j, 1);
  n.layers[l + 1].W.forEach((row) => row.splice(j, 1));
  return n;
}

/** Insert a hidden layer of `units` before layer index `at` (0..L-1). */
export function addLayer(net: Network, at: number, units: number, act: ActivationName, w0 = 0.5): Network {
  const n = cloneNet(net);
  const s = sizes(net);
  const fanIn = s[at];
  const newLayer: Layer = {
    W: Array.from({ length: units }, (_, j) => Array.from({ length: fanIn }, (_, i) => (i === j ? w0 : w0 / 2))),
    b: new Array(units).fill(0),
    act,
  };
  const next = n.layers[at];
  next.W = next.W.map((_, j) => Array.from({ length: units }, (_, i) => (i === j ? w0 : w0 / 2)));
  n.layers.splice(at, 0, newLayer);
  return n;
}

/** Remove hidden layer l, reconnecting its neighbours with fresh weights. */
export function removeLayer(net: Network, l: number, w0 = 0.5): Network {
  if (l >= net.layers.length - 1) return net;
  const n = cloneNet(net);
  const fanIn = sizes(net)[l];
  n.layers.splice(l, 1);
  n.layers[l].W = n.layers[l].W.map((_, j) => Array.from({ length: fanIn }, (_, i) => (i === j ? w0 : w0 / 2)));
  return n;
}
