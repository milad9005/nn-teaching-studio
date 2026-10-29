// The shared network workspace. Every network panel (canvas, inspector, equations, graphs,
// history) reads this one store; derived numbers come from useLabDerived(), never from copies.

import { create } from 'zustand';
import { useMemo } from 'react';
import {
  addLayer,
  addUnit,
  batchLossGrad,
  cloneNet,
  datasetLoss,
  flatten,
  initialize,
  loadPreset,
  maskFor,
  newTrainState,
  removeLayer,
  removeUnit,
  sampleResult,
  sizes,
  trainStep,
  validate,
  LIMITS,
  type ActivationName,
  type BatchConfig,
  type Dataset,
  type Grads,
  type InitName,
  type LossName,
  type LossOptions,
  type Network,
  type OptimizerConfig,
  type OptimizerName,
  type PresetId,
  type TrainConfig,
  type TrainState,
  type ParamRef,
} from '../engine';

/** Unit addresses use activation indices: layer 0 is the input, layer l+1 is the output of W[l]. */
export type Selection =
  | { type: 'unit'; layer: number; unit: number }
  | { type: 'weight'; layer: number; j: number; i: number }
  | { type: 'bias'; layer: number; j: number }
  | { type: 'loss' }
  | null;

/**
 * Step-by-step execution. `null` shows everything. While stepping forward, k counts revealed
 * activation layers (1 = input only ... L+1 = output, L+2 = loss). While stepping backward,
 * k counts revealed gradient layers from the loss (1 = dL/doutput, 2 = last layer's dz, ...).
 */
export type Stepper = null | { phase: 'forward' | 'backward'; k: number } | { phase: 'update'; k: 0 };

export interface HistoryPoint {
  iter: number;
  epoch: number;
  loss: number;
  val?: number;
  /** Flat parameters after the update (kept for trajectories on small nets). */
  theta?: number[];
  batch?: number[];
}

export interface LastUpdate {
  before: Network;
  grads: Grads;
  delta: number[];
  batch: number[];
  lr: number;
  optimizer: OptimizerName;
  clipped: boolean;
}

export type LabError = null | 'nonFiniteGrad' | 'nonFiniteParam' | 'diverged' | 'invalid' | 'tooLarge';

export interface LabState {
  presetId: PresetId | 'custom';
  net: Network;
  data: Dataset;
  val?: Dataset;
  loss: LossName;
  lossOpts: LossOptions;
  opt: OptimizerConfig;
  batch: BatchConfig;
  frozen: string[];
  units?: { x: string; y: string };
  sample: number;
  train: TrainState;
  history: HistoryPoint[];
  selection: Selection;
  stepper: Stepper;
  running: boolean;
  error: LabError;
  lastUpdate: LastUpdate | null;
  /** Bumped on every value change so animations can key off it. */
  version: number;

  loadPreset: (id: PresetId) => void;
  resetPreset: () => void;
  setNet: (net: Network, keepHistory?: boolean) => void;
  setParam: (p: Pick<ParamRef, 'kind' | 'layer' | 'j' | 'i'>, v: number) => void;
  setInput: (n: number, i: number, v: number) => void;
  setTarget: (n: number, k: number, v: number) => void;
  setData: (data: Dataset) => void;
  setActivation: (layer: number, act: ActivationName) => void;
  setLoss: (loss: LossName, opts?: LossOptions) => void;
  setOpt: (p: Partial<OptimizerConfig>) => void;
  setBatch: (p: Partial<BatchConfig>) => void;
  select: (s: Selection) => void;
  setSample: (n: number) => void;
  addUnit: (layer: number) => void;
  removeUnit: (layer: number, unit: number) => void;
  addLayer: () => void;
  removeLayer: (layer: number) => void;
  reinit: (init: InitName, seed?: number) => void;
  forwardStep: () => void;
  backwardStep: () => void;
  optimizerStep: () => boolean;
  showAll: () => void;
  trainSteps: (steps: number) => void;
  setRunning: (r: boolean) => void;
  resetTraining: () => void;
}

const trainCfg = (s: Pick<LabState, 'loss' | 'lossOpts' | 'opt' | 'batch' | 'net' | 'frozen'>): TrainConfig => ({
  loss: s.loss,
  lossOpts: s.lossOpts,
  opt: s.opt,
  batch: s.batch,
  mask: maskFor(s.net, s.frozen),
  divergeAt: 1e8,
});

const MAX_HISTORY = 5000;
const keepTheta = (net: Network) => flatten(net).length <= 80;

function historyStart(net: Network, data: Dataset, loss: LossName, lossOpts: LossOptions, val?: Dataset): HistoryPoint[] {
  return [
    {
      iter: 0,
      epoch: 0,
      loss: datasetLoss(net, data, loss, lossOpts),
      val: val ? datasetLoss(net, val, loss, lossOpts) : undefined,
      theta: keepTheta(net) ? flatten(net) : undefined,
    },
  ];
}

function fromPreset(id: PresetId) {
  const p = loadPreset(id);
  const base = { net: p.net, data: p.data, loss: p.loss, lossOpts: p.lossOpts, opt: p.opt, batch: p.batch, frozen: p.frozen };
  return {
    presetId: id,
    ...base,
    val: p.val,
    units: p.units,
    sample: 0,
    train: newTrainState(p.net, p.data, trainCfg(base)),
    history: historyStart(p.net, p.data, p.loss, p.lossOpts, p.val),
    selection: null,
    stepper: null,
    running: false,
    error: null,
    lastUpdate: null,
  } satisfies Partial<LabState>;
}

export const useLab = create<LabState>((set, get) => {
  /** Replace the network after an edit: keeps training state consistent with the new parameters. */
  const commitNet = (net: Network, extra: Partial<LabState> = {}, keepHistory = false) => {
    const s = get();
    const errs = validate(net);
    if (errs.some((e) => e.code === 'tooLarge')) return set({ error: 'tooLarge' });
    const cfg = trainCfg({ ...s, net, ...extra });
    const fresh = newTrainState(net, s.data, cfg);
    // A hand edit mid-run keeps the run going (same schedule and optimizer moments);
    // a new network starts a new run: iteration 0, fresh optimizer state, fresh history.
    const continueRun = keepHistory && flatten(net).length === flatten(s.net).length;
    set({
      net,
      train: continueRun ? { ...fresh, schedule: s.train.schedule, optState: s.train.optState } : fresh,
      history: keepHistory ? s.history : historyStart(net, s.data, s.loss, s.lossOpts, s.val),
      error: null,
      lastUpdate: null,
      version: s.version + 1,
      ...extra,
    });
  };

  return {
    ...fromPreset('chain-two'),
    version: 0,

    loadPreset: (id) => set((s) => ({ ...fromPreset(id), version: s.version + 1 })),
    resetPreset: () => {
      const id = get().presetId;
      if (id !== 'custom') get().loadPreset(id);
    },
    setNet: (net, keepHistory) => commitNet(net, {}, keepHistory),
    setParam: (p, v) => {
      if (!Number.isFinite(v)) return;
      const net = cloneNet(get().net);
      if (p.kind === 'W') net.layers[p.layer].W[p.j][p.i] = v;
      else net.layers[p.layer].b[p.j] = v;
      commitNet(net, {}, true);
    },
    setInput: (n, i, v) => {
      if (!Number.isFinite(v)) return;
      const s = get();
      const X = s.data.X.map((r) => r.slice());
      X[n][i] = v;
      set({ data: { ...s.data, X }, version: s.version + 1, lastUpdate: null });
    },
    setTarget: (n, k, v) => {
      if (!Number.isFinite(v)) return;
      const s = get();
      const Y = s.data.Y.map((r) => r.slice());
      Y[n][k] = v;
      set({ data: { ...s.data, Y }, version: s.version + 1, lastUpdate: null });
    },
    setData: (data) => {
      const s = get();
      if (validate(s.net, data).length) return set({ error: 'invalid' });
      set({
        data,
        sample: Math.min(s.sample, data.X.length - 1),
        train: newTrainState(s.net, data, trainCfg(s)),
        history: historyStart(s.net, data, s.loss, s.lossOpts, s.val),
        version: s.version + 1,
        error: null,
      });
    },
    setActivation: (layer, act) => {
      const net = cloneNet(get().net);
      net.layers[layer].act = act;
      commitNet(net, { presetId: 'custom' });
    },
    setLoss: (loss, opts) => {
      const s = get();
      set({
        loss,
        lossOpts: { ...s.lossOpts, ...opts },
        history: historyStart(s.net, s.data, loss, { ...s.lossOpts, ...opts }, s.val),
        version: s.version + 1,
      });
    },
    setOpt: (p) => {
      const s = get();
      const opt = { ...s.opt, ...p };
      // Changing the optimizer type restarts its internal state (velocity, moments).
      const optState = p.name && p.name !== s.opt.name ? newTrainState(s.net, s.data, trainCfg(s)).optState : s.train.optState;
      set({ opt, train: { ...s.train, optState }, version: s.version + 1 });
    },
    setBatch: (p) => {
      const s = get();
      const batch = { ...s.batch, ...p };
      // A new batching scheme restarts the schedule, so the history restarts with it.
      set({
        batch,
        train: newTrainState(s.net, s.data, trainCfg({ ...s, batch })),
        history: historyStart(s.net, s.data, s.loss, s.lossOpts, s.val),
        lastUpdate: null,
        version: s.version + 1,
      });
    },
    select: (selection) => set({ selection }),
    setSample: (sample) => set((s) => ({ sample: Math.max(0, Math.min(s.data.X.length - 1, sample)), version: s.version + 1 })),

    addUnit: (layer) => {
      const s = get();
      if (layer >= s.net.layers.length - 1 || s.net.layers[layer].b.length >= LIMITS.maxUnits) return;
      commitNet(addUnit(s.net, layer, 0.3), { presetId: 'custom', selection: null });
    },
    removeUnit: (layer, unit) => {
      const s = get();
      if (layer >= s.net.layers.length - 1) return;
      commitNet(removeUnit(s.net, layer, unit), { presetId: 'custom', selection: null });
    },
    addLayer: () => {
      const s = get();
      if (s.net.layers.length - 1 >= LIMITS.maxHiddenLayers) return;
      const L = s.net.layers.length;
      const act = L > 1 ? s.net.layers[L - 2].act : 'relu';
      const units = sizes(s.net)[L - 1] > 1 ? sizes(s.net)[L - 1] : 2;
      commitNet(addLayer(s.net, L - 1, units, act, 0.5), { presetId: 'custom', selection: null });
    },
    removeLayer: (layer) => {
      const s = get();
      if (layer >= s.net.layers.length - 1) return;
      commitNet(removeLayer(s.net, layer), { presetId: 'custom', selection: null });
    },
    reinit: (init, seed = 1) => {
      const s = get();
      commitNet(initialize(s.net, init, seed), {});
    },

    forwardStep: () => {
      const s = get();
      const L = s.net.layers.length;
      const st = s.stepper;
      const k = st && st.phase === 'forward' ? st.k + 1 : 1;
      set({ stepper: k > L + 2 ? null : { phase: 'forward', k } });
    },
    backwardStep: () => {
      const s = get();
      const L = s.net.layers.length;
      const st = s.stepper;
      const k = st && st.phase === 'backward' ? st.k + 1 : 1;
      set({ stepper: k > L + 1 ? null : { phase: 'backward', k } });
    },
    optimizerStep: () => {
      const s = get();
      const r = trainStep(s.train, s.data, trainCfg(s));
      if (!r.ok) {
        set({ error: r.error, running: false });
        return false;
      }
      const net = r.state.net;
      const point: HistoryPoint = {
        iter: r.state.schedule.iteration,
        epoch: r.state.schedule.epoch,
        loss: r.loss,
        val: s.val ? datasetLoss(net, s.val, s.loss, s.lossOpts) : undefined,
        theta: keepTheta(net) ? flatten(net) : undefined,
        batch: r.batch,
      };
      set({
        net,
        train: r.state,
        history: [...s.history, point].slice(-MAX_HISTORY),
        lastUpdate: {
          before: s.net,
          grads: r.grads,
          delta: r.delta,
          batch: r.batch,
          lr: s.opt.lr,
          optimizer: s.opt.name,
          clipped: r.clipped,
        },
        stepper: s.running ? null : { phase: 'update', k: 0 },
        error: null,
        version: s.version + 1,
      });
      return true;
    },
    showAll: () => set({ stepper: null }),
    trainSteps: (steps) => {
      for (let i = 0; i < steps; i++) if (!get().optimizerStep()) return;
    },
    setRunning: (running) => set({ running, stepper: null }),
    resetTraining: () => {
      const s = get();
      if (s.presetId !== 'custom') return s.loadPreset(s.presetId);
      set({
        train: newTrainState(s.net, s.data, trainCfg(s)),
        history: historyStart(s.net, s.data, s.loss, s.lossOpts, s.val),
        running: false,
        error: null,
        lastUpdate: null,
      });
    },
  };
});

export const labTrainCfg = trainCfg;

/** Everything computed from the current lab state. Recomputed only when inputs change. */
export function useLabDerived() {
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const loss = useLab((s) => s.loss);
  const lossOpts = useLab((s) => s.lossOpts);
  const sample = useLab((s) => s.sample);
  const batchCfg = useLab((s) => s.batch);
  const schedule = useLab((s) => s.train.schedule);
  return useMemo(() => {
    const n = Math.min(sample, data.X.length - 1);
    const one = sampleResult(net, data.X[n], data.Y[n], loss, lossOpts);
    const full = batchLossGrad(net, data, loss, lossOpts);
    const B = batchCfg.size <= 0 || batchCfg.size >= data.X.length ? data.X.length : batchCfg.size;
    const nextIdx = schedule.order.slice(schedule.cursor, schedule.cursor + B);
    const next = batchLossGrad(net, data, loss, lossOpts, nextIdx);
    return { sample: n, one, full, next, nextIdx, L: net.layers.length, sizes: sizes(net) };
  }, [net, data, loss, lossOpts, sample, batchCfg, schedule]);
}

/** What the stepper allows to be shown. */
export function visibility(stepper: Stepper, L: number) {
  if (!stepper || stepper.phase === 'update') return { act: (_l: number) => true, loss: true, grad: (_l: number) => true, dOut: true };
  if (stepper.phase === 'forward')
    return { act: (l: number) => l < stepper.k, loss: stepper.k > L + 1, grad: () => false, dOut: false };
  // backward: k = 1 -> dL/doutput; k = 2 -> dz of layer L-1 (activation index L); ...
  return {
    act: () => true,
    loss: true,
    dOut: stepper.k >= 1,
    grad: (actLayer: number) => actLayer >= 1 && L + 1 - actLayer < stepper.k,
  };
}
