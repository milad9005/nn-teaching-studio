// Lesson 10 math (data): one neuron ŷ = w·x + b learning a cloud of examples with the 'mini-batch'
// preset (8 examples, batches of 2, plain steps). Every slope comes from the engine's backprop and
// every training step is the engine's trainStep, so the scene only draws what is computed here.

import {
  batchLossGrad,
  datasetLoss,
  effectiveBatch,
  epochOrder,
  loadPreset,
  newTrainState,
  trainStep,
  type Dataset,
  type Network,
  type TrainConfig,
} from '../../engine';

/** Knobs of the one-neuron machine. */
export interface Knobs {
  w: number;
  b: number;
}

export interface Arrow {
  /** Which example (index into the data). */
  n: number;
  /** Slopes of this example's loss: [dL/dw, dL/db]. */
  g: [number, number];
  /** The step this slope asks for: −η · slope, in knob space (w, b). */
  step: [number, number];
}

export interface Frame extends Knobs {
  /** Update number (0 = before training). */
  iter: number;
  /** Epoch the *next* batch belongs to (1-based). */
  epoch: number;
  /** Loss on all the examples at these knobs. */
  loss: number;
  /** Loss on the batch used by the update that led here, measured just before it (what a training log shows). */
  batchLoss: number;
  /** Examples used by the update that led here (empty for the start). */
  batch: number[];
}

export interface DataStory {
  data: Dataset;
  lr: number;
  B: number;
  /** Batches per epoch. */
  perEpoch: number;
  start: Knobs;
  /** Each example's own slope and step at the start. */
  arrows: Arrow[];
  /** The average of all the example steps. */
  mean: [number, number];
  /** The first batch (shuffled order of epoch 1) and the average step of only its examples. */
  firstBatch: number[];
  batchMean: [number, number];
  /** The batches of epoch 1, in order. */
  epoch1: number[][];
  /** Training history, one frame per update, frames[0] = start. */
  frames: Frame[];
  /** The best line through all the dots (least squares), for reference. */
  best: Knobs;
}

const knobsOf = (net: Network): Knobs => ({ w: net.layers[0].W[0][0], b: net.layers[0].b[0] });

/** Mean of [a, b] pairs. */
export const meanOf = (vs: [number, number][]): [number, number] => [vs.reduce((s, v) => s + v[0], 0) / vs.length, vs.reduce((s, v) => s + v[1], 0) / vs.length];

/** Least-squares line through the data (the knobs where the loss on all examples is lowest). */
export function bestLine(data: Dataset): Knobs {
  const xs = data.X.map((r) => r[0]);
  const ys = data.Y.map((r) => r[0]);
  const N = xs.length;
  const mx = xs.reduce((s, v) => s + v, 0) / N;
  const my = ys.reduce((s, v) => s + v, 0) / N;
  const sxy = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0);
  const sxx = xs.reduce((s, x) => s + (x - mx) ** 2, 0);
  const w = sxy / sxx;
  return { w, b: my - w * mx };
}

/**
 * The lesson steps a little more gently than the preset (0.03 instead of 0.1), so training takes a
 * few epochs and the bumps of each batch stay visible on the way down.
 */
export const LESSON_RATE = 0.03;

let cache: DataStory | null = null;

/** The whole lesson, computed once. `epochs` full passes of training are recorded. */
export function dataStory(epochs = 12): DataStory {
  if (cache && cache.frames.length === epochs * cache.perEpoch + 1) return cache;
  const p = loadPreset('mini-batch');
  const N = p.data.X.length;
  const B = effectiveBatch(N, p.batch);
  const perEpoch = Math.ceil(N / B);
  const lr = LESSON_RATE;
  const arrows: Arrow[] = p.data.X.map((_, n) => {
    const g = batchLossGrad(p.net, p.data, p.loss, p.lossOpts, [n]).grads;
    const gw = g.W[0][0][0];
    const gb = g.b[0][0];
    return { n, g: [gw, gb], step: [-lr * gw, -lr * gb] };
  });
  const order = epochOrder(N, p.batch, 0);
  const epoch1 = Array.from({ length: perEpoch }, (_, k) => order.slice(k * B, k * B + B));
  const firstBatch = epoch1[0];
  const cfg: TrainConfig = { loss: p.loss, lossOpts: p.lossOpts, opt: { ...p.opt, lr }, batch: p.batch, divergeAt: 1e8 };
  let st = newTrainState(p.net, p.data, cfg);
  const frames: Frame[] = [{ ...knobsOf(p.net), iter: 0, epoch: 1, loss: datasetLoss(p.net, p.data, p.loss, p.lossOpts), batchLoss: NaN, batch: [] }];
  for (let i = 0; i < epochs * perEpoch; i++) {
    const r = trainStep(st, p.data, cfg);
    if (!r.ok) break;
    st = r.state;
    frames.push({ ...knobsOf(st.net), iter: i + 1, epoch: st.schedule.epoch + 1, loss: r.loss, batchLoss: r.batchLoss, batch: r.batch });
  }
  cache = {
    data: p.data,
    lr,
    B,
    perEpoch,
    start: knobsOf(p.net),
    arrows,
    mean: meanOf(arrows.map((a) => a.step)),
    firstBatch,
    batchMean: meanOf(firstBatch.map((n) => arrows[n].step)),
    epoch1,
    frames,
    best: bestLine(p.data),
  };
  return cache;
}

/** Angle in degrees between two directions. */
export function angleBetween(a: [number, number], b: [number, number]): number {
  const d = (a[0] * b[0] + a[1] * b[1]) / (Math.hypot(...a) * Math.hypot(...b));
  return (Math.acos(Math.max(-1, Math.min(1, d))) * 180) / Math.PI;
}

/** Loss on all examples at knobs (w, b). */
export function lossAt(data: Dataset, w: number, b: number): number {
  return data.X.reduce((s, x, n) => s + (w * x[0] + b - data.Y[n][0]) ** 2, 0) / data.X.length;
}

/**
 * The contour line where the loss on all examples equals `level`. The loss is a bowl around the best
 * knobs m: L(m + r·d) = L(m) + r²·(d·H·d) with H = [[mean x², mean x], [mean x, 1]], so every
 * direction d gives its radius directly. Empty when `level` is below the bottom of the bowl.
 */
export function lossContour(data: Dataset, level: number, n = 96): [number, number][] {
  const m = bestLine(data);
  const L0 = lossAt(data, m.w, m.b);
  if (level <= L0) return [];
  const xs = data.X.map((r) => r[0]);
  const xx = xs.reduce((s, x) => s + x * x, 0) / xs.length;
  const x1 = xs.reduce((s, x) => s + x, 0) / xs.length;
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = (2 * Math.PI * i) / n;
    const d = [Math.cos(t), Math.sin(t)];
    const q = xx * d[0] * d[0] + 2 * x1 * d[0] * d[1] + d[1] * d[1];
    const r = Math.sqrt((level - L0) / q);
    return [m.w + r * d[0], m.b + r * d[1]];
  });
}
