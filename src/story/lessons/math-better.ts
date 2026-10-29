// Lesson 11 math (better): a narrow valley with two knobs p and q, walked by plain steps, momentum
// and Adam (every step is the engine's optimizerStep, the same code that trains networks), and a tiny
// 1 → 4 → 1 ReLU network trained from two starts to show why the starting knobs must differ.

import {
  datasetLoss,
  defaultOptimizer,
  initOptState,
  initialize,
  linspace,
  makeNetwork,
  newTrainState,
  optimizerStep,
  trainStep,
  type Dataset,
  type Network,
  type OptimizerConfig,
  type OptimizerName,
  type TrainConfig,
} from '../../engine';

// ---------------------------------------------------------------- the valley

/** L(p, q) = ½(A·p² + Bq·q²): a gentle floor along p, steep walls across q. */
export const VALLEY = { A: 0.1, B: 10 };
export const valleyLoss = (p: number, q: number) => 0.5 * (VALLEY.A * p * p + VALLEY.B * q * q);
export const valleySlope = (p: number, q: number): [number, number] => [VALLEY.A * p, VALLEY.B * q];
export const START: [number, number] = [-4.5, 1.5];
export const P_RANGE: [number, number] = [-5.2, 1.4];
export const Q_RANGE: [number, number] = [-2, 2];
/** Momentum keeps this share of the previous step. */
export const BETA = 0.9;
export const RATE: Record<OptimizerName, number> = { sgd: 0.19, momentum: 0.04, adam: 0.3 };
export const RACE_STEPS = 60;
export const OPTS: OptimizerName[] = ['sgd', 'momentum', 'adam'];

/** Semi-axes of the contour ellipse where the loss equals `level`: half-width along p and along q. */
export const ellipseAt = (level: number): [number, number] => [Math.sqrt((2 * level) / VALLEY.A), Math.sqrt((2 * level) / VALLEY.B)];

export interface Path {
  /** points[k] = knobs (p, q) after k steps. */
  points: [number, number][];
  losses: number[];
  /** slopes[k] = slope at points[k], used by step k → k+1. */
  slopes: [number, number][];
  /** steps[k] = the move from points[k] to points[k+1]. */
  steps: [number, number][];
}

export const configFor = (name: OptimizerName): OptimizerConfig => ({ ...defaultOptimizer(name, RATE[name]), momentum: BETA });

/** Walk the valley with one optimizer; every move is the engine's optimizerStep. */
export function walk(name: OptimizerName, steps = RACE_STEPS): Path {
  const cfg = configFor(name);
  let theta: number[] = START.slice();
  let st = initOptState(2);
  const points: [number, number][] = [[theta[0], theta[1]]];
  const losses = [valleyLoss(theta[0], theta[1])];
  const slopes: [number, number][] = [];
  const moves: [number, number][] = [];
  for (let i = 0; i < steps; i++) {
    const g = valleySlope(theta[0], theta[1]);
    const r = optimizerStep(theta, g, cfg, st);
    if (!r.ok) break;
    slopes.push(g);
    moves.push([r.delta[0], r.delta[1]]);
    theta = r.theta;
    st = r.state;
    points.push([theta[0], theta[1]]);
    losses.push(valleyLoss(theta[0], theta[1]));
  }
  return { points, losses, slopes, steps: moves };
}

const pathCache = new Map<OptimizerName, Path>();
export const pathOf = (name: OptimizerName): Path => {
  let p = pathCache.get(name);
  if (!p) pathCache.set(name, (p = walk(name)));
  return p;
};

/** How many times a path turns back across the valley (sign changes of its q-moves). */
export function zigzags(path: Path): number {
  let n = 0;
  for (let k = 1; k < path.steps.length; k++) if (Math.sign(path.steps[k][1]) !== Math.sign(path.steps[k - 1][1]) && Math.abs(path.steps[k][1]) > 1e-9) n++;
  return n;
}

/** Total distance travelled. */
export const pathLength = (path: Path) => path.steps.reduce((s, d) => s + Math.hypot(d[0], d[1]), 0);

/** First step at which the loss falls below `level` (−1 if never). */
export const stepsToReach = (path: Path, level: number) => path.losses.findIndex((L) => L < level);

export interface MomentumParts {
  /** Index of the new step (the move from points[k] to points[k+1]). */
  k: number;
  at: [number, number];
  prev: [number, number];
  /** BETA × previous step. */
  kept: [number, number];
  /** −rate × the new slope. */
  push: [number, number];
  slope: [number, number];
  /** kept + push: the new step (equals the engine's move). */
  next: [number, number];
}

/**
 * Split momentum step k into "0.9 × previous step" and "the new slope's push". The engine stores
 * v ← 0.9·v + slope and moves by −rate·v, so step k = 0.9 · step k−1 − rate · slope k.
 */
export function momentumParts(k: number): MomentumParts {
  const path = pathOf('momentum');
  const prev = path.steps[k - 1];
  const slope = path.slopes[k];
  const kept: [number, number] = [BETA * prev[0], BETA * prev[1]];
  const push: [number, number] = [-RATE.momentum * slope[0], -RATE.momentum * slope[1]];
  return { k, at: path.points[k], prev, kept, push, slope, next: path.steps[k] };
}

/**
 * The momentum step that shows the idea best: the kept part and the new push are both big and point
 * in clearly different directions (largest sin(angle) × the smaller of the two lengths).
 */
export function showcaseStep(maxK = 30): number {
  const path = pathOf('momentum');
  let best = 1;
  let score = -Infinity;
  for (let k = 1; k < Math.min(maxK, path.steps.length); k++) {
    const m = momentumParts(k);
    const n1 = Math.hypot(...m.kept);
    const n2 = Math.hypot(...m.push);
    const sin = Math.abs(m.kept[0] * m.push[1] - m.kept[1] * m.push[0]) / (n1 * n2);
    const s = Math.min(n1, n2) * sin;
    if (s > score) {
      score = s;
      best = k;
    }
  }
  return best;
}

// ---------------------------------------------------------------- the start

/** The data: 9 dots on a wave. */
export const wave = (x: number) => Math.sin(1.5 * x);
export const WAVE: Dataset = (() => {
  const xs = linspace(-2, 2, 9);
  return { X: xs.map((x) => [x]), Y: xs.map((x) => [wave(x)]) };
})();
export const HIDDEN = 4;
/** Every knob of the "same start". */
export const SAME = 0.5;
/** Range of the random start: every weight between −RANDOM and +RANDOM. */
export const RANDOM = 0.5;
export const SEED = 8;
export const INIT_RATE = 0.05;
export const INIT_STEPS = 200;

export type StartId = 'same' | 'random';

export const startNet = (start: StartId): Network =>
  start === 'same'
    ? initialize(makeNetwork([1, HIDDEN, 1], ['relu', 'linear']), 'constant', 1, SAME)
    : initialize(makeNetwork([1, HIDDEN, 1], ['relu', 'linear']), 'uniform', SEED, RANDOM);

export interface Unit {
  /** Input weight, bias and weight out of one hidden neuron. */
  w: number;
  b: number;
  u: number;
}

export interface Run {
  /** units[k][j] = hidden neuron j after k steps. */
  units: Unit[][];
  outB: number[];
  loss: number[];
}

const unitsOf = (net: Network): Unit[] => net.layers[0].W.map((row, j) => ({ w: row[0], b: net.layers[0].b[j], u: net.layers[1].W[0][j] }));

/** Train the tiny network from a start; every step is the engine's trainStep (Adam, all 9 dots). */
export function trainFrom(start: StartId, steps = INIT_STEPS): Run {
  const net = startNet(start);
  const cfg: TrainConfig = { loss: 'mse', lossOpts: {}, opt: defaultOptimizer('adam', INIT_RATE), batch: { size: 0, shuffle: false, seed: 1 }, divergeAt: 1e8 };
  let st = newTrainState(net, WAVE, cfg);
  const units = [unitsOf(net)];
  const outB = [net.layers[1].b[0]];
  const loss = [datasetLoss(net, WAVE, 'mse')];
  for (let i = 0; i < steps; i++) {
    const r = trainStep(st, WAVE, cfg);
    if (!r.ok) break;
    st = r.state;
    units.push(unitsOf(st.net));
    outB.push(st.net.layers[1].b[0]);
    loss.push(r.loss);
  }
  return { units, outB, loss };
}

const runCache = new Map<StartId, Run>();
export const runOf = (start: StartId): Run => {
  let r = runCache.get(start);
  if (!r) runCache.set(start, (r = trainFrom(start)));
  return r;
};

/** One hidden neuron's piece of the output: u · ReLU(w·x + b). */
export const piece = (n: Unit, x: number) => n.u * Math.max(0, n.w * x + n.b);
/** The network's output: the sum of the pieces plus the output bias. */
export const output = (units: Unit[], outB: number, x: number) => units.reduce((s, n) => s + piece(n, x), outB);

/** How many different neurons there are (closer than `tol` in every number counts as the same). */
export function distinctUnits(units: Unit[], tol = 1e-9): number {
  const reps: Unit[] = [];
  for (const n of units) if (!reps.some((q) => Math.abs(n.w - q.w) <= tol && Math.abs(n.b - q.b) <= tol && Math.abs(n.u - q.u) <= tol)) reps.push(n);
  return reps.length;
}
