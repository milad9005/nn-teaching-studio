// Lesson 16 math: training the same small network from different initializations, and measuring
// how activation spread changes with depth. Weights come from the engine's initialize().

import {
  defaultOptimizer,
  forward,
  initialize,
  linspace,
  makeNetwork,
  mulberry32,
  newTrainState,
  normal,
  trainStep,
  datasetLoss,
  type ActivationName,
  type Dataset,
  type InitName,
  type Network,
  type TrainConfig,
} from '../../engine';

/** Target for the symmetry experiment: y = sin(1.5 x) on 9 points. */
export const SYM_DATA: Dataset = (() => {
  const xs = linspace(-2, 2, 9);
  return { X: xs.map((x) => [x]), Y: xs.map((x) => [Math.sin(1.5 * x)]) };
})();

export const HIDDEN = 4;

export const symNet = (init: InitName, seed: number, scale: number): Network =>
  initialize(makeNetwork([1, HIDDEN, 1], ['tanh', 'linear']), init, seed, scale);

export interface Unit {
  /** Input weight, bias and outgoing weight of a hidden unit. */
  w: number;
  b: number;
  v: number;
}

export interface InitRun {
  loss: number[];
  /** units[k][j] = hidden unit j after k steps. */
  units: Unit[][];
  nets: Network[];
  failed: boolean;
}

const unitsOf = (net: Network): Unit[] =>
  net.layers[0].W.map((row, j) => ({ w: row[0], b: net.layers[0].b[j], v: net.layers[1].W[0][j] }));

export function trainFrom(net: Network, data: Dataset, steps: number, lr = 0.05): InitRun {
  const cfg: TrainConfig = {
    loss: 'mse',
    lossOpts: {},
    opt: defaultOptimizer('adam', lr),
    batch: { size: 0, shuffle: false, seed: 1 },
    divergeAt: 1e8,
  };
  let st = newTrainState(net, data, cfg);
  const loss = [datasetLoss(net, data, 'mse')];
  const units = [unitsOf(net)];
  const nets = [net];
  for (let i = 0; i < steps; i++) {
    const r = trainStep(st, data, cfg);
    if (!r.ok) return { loss, units, nets, failed: true };
    st = r.state;
    loss.push(r.loss);
    units.push(unitsOf(st.net));
    nets.push(st.net);
  }
  return { loss, units, nets, failed: false };
}

/** Largest difference between any two hidden units (0 = perfectly symmetric copies). */
export function unitSpread(units: Unit[]): number {
  let m = 0;
  for (const u of units)
    for (const q of units) m = Math.max(m, Math.abs(u.w - q.w), Math.abs(u.b - q.b), Math.abs(u.v - q.v));
  return m;
}

// ---------------------------------------------------------------------------
// Depth experiment: a deep stack fed with random input vectors.

export const DEEP = { width: 16, depth: 6, samples: 300 };

export type DeepInit = 'small' | 'large' | 'xavier' | 'he';
export const DEEP_INITS: DeepInit[] = ['small', 'large', 'xavier', 'he'];

/** small = U(-0.1, 0.1), large = U(-1.5, 1.5). */
export const deepNet = (act: ActivationName, init: DeepInit, seed = 7): Network => {
  const name: InitName = init === 'small' || init === 'large' ? 'uniform' : init;
  return initialize(makeNetwork([DEEP.width, ...new Array(DEEP.depth).fill(DEEP.width)], new Array(DEEP.depth).fill(act)), name, seed, init === 'small' ? 0.1 : 1.5);
};

export interface LayerStats {
  std: number;
  /** Share of activations with |a| > 0.95 (tanh saturation) or exactly 0 (dead ReLU). */
  extreme: number;
  hist: { lo: number; hi: number; counts: number[] };
}

export function deepStats(act: ActivationName, init: DeepInit, seed = 7, bins = 24): LayerStats[] {
  const net = deepNet(act, init, seed);
  const rng = mulberry32(5);
  const inputs = Array.from({ length: DEEP.samples }, () => Array.from({ length: DEEP.width }, () => normal(rng)));
  const traces = inputs.map((x) => forward(net, x));
  return net.layers.map((_, l) => {
    const vals = traces.flatMap((t) => t.a[l + 1]);
    const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
    const std = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length);
    const extreme = vals.filter((v) => (act === 'relu' ? v === 0 : Math.abs(v) > 0.95)).length / vals.length;
    let lo = act === 'tanh' ? -1 : 0;
    let hi = act === 'tanh' ? 1 : Math.max(1e-6, ...vals);
    if (act !== 'tanh' && hi < 1e-6) hi = 1e-6;
    if (lo === hi) lo = hi - 1;
    const counts = new Array(bins).fill(0);
    vals.forEach((v) => {
      const k = Math.min(bins - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * bins)));
      counts[k] += 1 / vals.length;
    });
    return { std, extreme, hist: { lo, hi, counts } };
  });
}

/** Xavier/Glorot uniform range and He normal std for a layer. */
export const xavierR = (fanIn: number, fanOut: number) => Math.sqrt(6 / (fanIn + fanOut));
export const heStd = (fanIn: number) => Math.sqrt(2 / fanIn);
