// The training loop: batching schedule, one training iteration, and divergence guards.

import { mulberry32, shuffled } from './rng';
import {
  batchLossGrad,
  datasetLoss,
  flatten,
  flattenGrads,
  unflatten,
  validate,
  type Dataset,
  type Grads,
  type Network,
} from './network';
import { initOptState, optimizerStep, type OptimizerConfig, type OptimizerState } from './optimizers';
import type { LossName, LossOptions } from './losses';

export interface BatchConfig {
  /** 0 or >= N means full batch. */
  size: number;
  shuffle: boolean;
  seed: number;
}

export interface Schedule {
  epoch: number;
  /** Iterations (parameter updates) done so far. */
  iteration: number;
  /** Position in `order` of the next sample. */
  cursor: number;
  order: number[];
}

export const newSchedule = (N: number, cfg: BatchConfig): Schedule => ({
  epoch: 0,
  iteration: 0,
  cursor: 0,
  order: epochOrder(N, cfg, 0),
});

export function epochOrder(N: number, cfg: BatchConfig, epoch: number): number[] {
  const ids = Array.from({ length: N }, (_, i) => i);
  return cfg.shuffle ? shuffled(ids, mulberry32(cfg.seed * 7919 + epoch)) : ids;
}

export const effectiveBatch = (N: number, cfg: BatchConfig) => (cfg.size <= 0 || cfg.size >= N ? N : cfg.size);

/** The samples the next update will use, without advancing the schedule. */
export function nextBatch(N: number, cfg: BatchConfig, s: Schedule): number[] {
  const B = effectiveBatch(N, cfg);
  return s.order.slice(s.cursor, Math.min(N, s.cursor + B));
}

export function advance(N: number, cfg: BatchConfig, s: Schedule): Schedule {
  const B = effectiveBatch(N, cfg);
  const cursor = s.cursor + B;
  if (cursor >= N) {
    const epoch = s.epoch + 1;
    return { epoch, iteration: s.iteration + 1, cursor: 0, order: epochOrder(N, cfg, epoch) };
  }
  return { ...s, iteration: s.iteration + 1, cursor };
}

export interface TrainConfig {
  loss: LossName;
  lossOpts: LossOptions;
  opt: OptimizerConfig;
  batch: BatchConfig;
  /** Frozen parameters (by flat index) are never updated. */
  mask?: boolean[];
  /** Stop when the loss exceeds this (divergence guard). */
  divergeAt: number;
}

export interface TrainState {
  net: Network;
  optState: OptimizerState;
  schedule: Schedule;
}

export const newTrainState = (net: Network, data: Dataset, cfg: TrainConfig): TrainState => ({
  net,
  optState: initOptState(flatten(net).length),
  schedule: newSchedule(data.X.length, cfg.batch),
});

export type StepOutcome =
  | {
      ok: true;
      state: TrainState;
      batch: number[];
      /** Mean loss on the batch before the update. */
      batchLoss: number;
      grads: Grads;
      delta: number[];
      clipped: boolean;
      /** Loss on the whole dataset after the update. */
      loss: number;
    }
  | { ok: false; error: 'nonFiniteGrad' | 'nonFiniteParam' | 'diverged' | 'invalid' };

export function trainStep(st: TrainState, data: Dataset, cfg: TrainConfig): StepOutcome {
  if (validate(st.net, data).length) return { ok: false, error: 'invalid' };
  const N = data.X.length;
  const batch = nextBatch(N, cfg.batch, st.schedule);
  const { loss: batchLoss, grads } = batchLossGrad(st.net, data, cfg.loss, cfg.lossOpts, batch);
  const r = optimizerStep(flatten(st.net), flattenGrads(st.net, grads), cfg.opt, st.optState, cfg.mask);
  if (!r.ok) return r;
  const net = unflatten(st.net, r.theta);
  const loss = datasetLoss(net, data, cfg.loss, cfg.lossOpts);
  if (!Number.isFinite(loss) || loss > cfg.divergeAt) return { ok: false, error: 'diverged' };
  return {
    ok: true,
    state: { net, optState: r.state, schedule: advance(N, cfg.batch, st.schedule) },
    batch,
    batchLoss,
    grads,
    delta: r.delta,
    clipped: r.clipped,
    loss,
  };
}
