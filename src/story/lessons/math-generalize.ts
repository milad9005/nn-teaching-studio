// Lesson 12 math (generalize): the 'overfit' preset (8 noisy dots from a smooth true curve, 14 dots
// set aside to check, a 1 → 8 → 1 network) trained with the engine. Records both losses at every
// step, the step where the check loss is lowest, and the networks at that step and at the end.

import { datasetLoss, linspace, loadPreset, mulberry32, newTrainState, normal, overfitTruth, predict, trainStep, type Dataset, type Network, type TrainConfig } from '../../engine';

export const TRAIN_STEPS = 600;

export interface FitStory {
  data: Dataset;
  /** Data set aside to check (never trained on). */
  val: Dataset;
  truth: (x: number) => number;
  /** trainLoss[k], valLoss[k] = losses after k steps. */
  trainLoss: number[];
  valLoss: number[];
  /** nets[k] = the network after k·SNAP steps (for animating the curve). */
  nets: Network[];
  snap: number;
  /** Step with the lowest check loss: where to stop. */
  stop: number;
  stopNet: Network;
  endNet: Network;
  /** How many numbers (knobs) the network has. */
  knobs: number;
}

/** Index of the smallest value. */
export function argMin(v: number[]): number {
  let k = 0;
  v.forEach((x, i) => {
    if (x < v[k]) k = i;
  });
  return k;
}

/** Mean squared distance between the true curve and the noisy answers: the wobble nobody can predict. */
export const noiseLoss = (d: Dataset, truth: (x: number) => number) => d.X.reduce((s, x, n) => s + (truth(x[0]) - d.Y[n][0]) ** 2, 0) / d.X.length;

/**
 * The preset's check dots, rebuilt with a fresh random generator. The preset draws them from one
 * generator shared by every loadPreset('overfit') call, so they change each time the preset is
 * loaded (the lab loads it when the lesson opens). Rebuilding them here keeps the lesson identical
 * every time: these are exactly the dots of the first load (same x, same seed 9, same wobble 0.25).
 */
export function checkDots(): Dataset {
  const rng = mulberry32(9);
  const xs = linspace(-1.95, 1.95, 14);
  return { X: xs.map((x) => [x]), Y: xs.map((x) => [overfitTruth(x) + normal(rng, 0, 0.25)]) };
}

let cache: FitStory | null = null;

export function fitStory(steps = TRAIN_STEPS, snap = 10): FitStory {
  if (cache && cache.trainLoss.length === steps + 1 && cache.snap === snap) return cache;
  const p = loadPreset('overfit');
  const val = checkDots();
  const cfg: TrainConfig = { loss: p.loss, lossOpts: p.lossOpts, opt: p.opt, batch: p.batch, divergeAt: 1e8 };
  let st = newTrainState(p.net, p.data, cfg);
  const trainLoss = [datasetLoss(p.net, p.data, p.loss)];
  const valLoss = [datasetLoss(p.net, val, p.loss)];
  const all: Network[] = [p.net];
  for (let i = 0; i < steps; i++) {
    const r = trainStep(st, p.data, cfg);
    if (!r.ok) break;
    st = r.state;
    all.push(st.net);
    trainLoss.push(r.loss);
    valLoss.push(datasetLoss(st.net, val, p.loss));
  }
  const stop = argMin(valLoss);
  const knobs = p.net.layers.reduce((s, l) => s + l.b.length * (l.W[0].length + 1), 0);
  cache = {
    data: p.data,
    val,
    truth: overfitTruth,
    trainLoss,
    valLoss,
    nets: all.filter((_, i) => i % snap === 0),
    snap,
    stop,
    stopNet: all[stop],
    endNet: all[all.length - 1],
    knobs,
  };
  return cache;
}

/** The network as a plain function of one number. */
export const asFn = (net: Network) => (x: number) => predict(net, [x])[0];

/** New inputs for the "use it" step: not among the training or check dots. */
export const NEW_INPUTS = [0.5, -1.2, 1.5];

/** Sizes of the preset, without training: how many knobs the network has and how many dots it learns. */
export function presetSizes() {
  const p = loadPreset('overfit');
  return { knobs: p.net.layers.reduce((s, l) => s + l.b.length * (l.W[0].length + 1), 0), dots: p.data.X.length, check: p.val!.X.length };
}
