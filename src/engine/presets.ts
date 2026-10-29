// Named, reproducible teaching scenarios. Text (names, outcomes) lives in src/i18n/presets.ts;
// the numbers here are verified by src/engine/presets.test.ts.

import { initialize } from './init';
import { linspace } from './numeric';
import { makeNetwork, paramRefs, type Dataset, type Network } from './network';
import { defaultOptimizer, type OptimizerConfig } from './optimizers';
import type { LossName, LossOptions } from './losses';
import type { BatchConfig } from './training';
import { mulberry32, normal } from './rng';

export type PresetId =
  | 'single-affine'
  | 'relu-threshold'
  | 'one-vs-many'
  | 'temperature'
  | 'single-weight'
  | 'two-weight'
  | 'chain-two'
  | 'branching'
  | 'gd-step'
  | 'lr-compare'
  | 'mini-batch'
  | 'overfit';

export interface Preset {
  id: PresetId;
  net: Network;
  data: Dataset;
  /** Held-out data (overfitting demo). */
  val?: Dataset;
  loss: LossName;
  lossOpts: LossOptions;
  opt: OptimizerConfig;
  batch: BatchConfig;
  /** Parameter ids that never train (e.g. a bias held at 0 to keep a landscape 2-D). */
  frozen: string[];
  /** Axis labels for the input and output, as i18n keys. */
  units?: { x: string; y: string };
}

const fullBatch: BatchConfig = { size: 0, shuffle: false, seed: 1 };

const base = (p: Partial<Preset> & Pick<Preset, 'id' | 'net' | 'data'>): Preset => ({
  loss: 'mse',
  lossOpts: { delta: 1 },
  opt: defaultOptimizer('sgd', 0.05),
  batch: fullBatch,
  frozen: [],
  ...p,
});

const overfitX = [-1.9, -1.4, -0.9, -0.35, 0.2, 0.7, 1.25, 1.8];
const overfitNoise = [0.25, -0.3, 0.2, -0.25, 0.3, -0.2, 0.25, -0.3];
const truth = (x: number) => Math.sin((Math.PI * x) / 2);
const valRng = mulberry32(9);
const valX = linspace(-1.95, 1.95, 14);

export const PRESETS: Record<PresetId, () => Preset> = {
  'single-affine': () =>
    base({
      id: 'single-affine',
      net: makeNetwork([1, 1], ['linear'], () => 2, () => 1),
      data: { X: [[1.5]], Y: [[5]] },
    }),
  'relu-threshold': () =>
    base({
      id: 'relu-threshold',
      net: makeNetwork([1, 1], ['relu'], () => 1, () => -1),
      data: { X: [[-1], [0.5], [1], [2]], Y: [[0], [0.2], [0], [1.5]] },
    }),
  'one-vs-many': () =>
    base({
      id: 'one-vs-many',
      net: {
        inputSize: 1,
        layers: [
          { W: [[1], [-1], [1], [-1]], b: [0, 0, -1, -1], act: 'relu' },
          { W: [[0.5, 0.5, 0.5, 0.5]], b: [0], act: 'linear' },
        ],
      },
      data: { X: linspace(-2, 2, 9).map((x) => [x]), Y: linspace(-2, 2, 9).map((x) => [0.5 * x * x]) },
      opt: defaultOptimizer('adam', 0.03),
    }),
  temperature: () =>
    base({
      id: 'temperature',
      net: makeNetwork([1, 1], ['linear'], () => 1, () => -1.2),
      data: { X: [[2.0], [2.5], [3.0], [3.5]], Y: [[0.6], [1.1], [1.9], [2.6]] },
      units: { x: 'units.outsideTemp', y: 'units.acPower' },
    }),
  'single-weight': () =>
    base({
      id: 'single-weight',
      net: makeNetwork([1, 1], ['linear'], () => 3, () => 0),
      data: { X: [[2]], Y: [[10]] },
      frozen: ['b1[0]'],
    }),
  'two-weight': () =>
    base({
      id: 'two-weight',
      net: { inputSize: 2, layers: [{ W: [[0.5, -1]], b: [0], act: 'linear' }] },
      data: { X: [[1, 0], [0, 1], [1, 1]], Y: [[2], [1], [3]] },
      frozen: ['b1[0]'],
      opt: defaultOptimizer('sgd', 0.1),
    }),
  'chain-two': () =>
    base({
      id: 'chain-two',
      net: makeNetwork([1, 1, 1], ['relu', 'relu'], (l) => (l === 0 ? 3 : 2), () => 0),
      data: { X: [[2]], Y: [[10]] },
      opt: defaultOptimizer('sgd', 0.01),
    }),
  branching: () =>
    base({
      id: 'branching',
      net: {
        inputSize: 1,
        layers: [
          { W: [[1], [0.5]], b: [0, 0], act: 'relu' },
          { W: [[1, 1], [0.5, -1]], b: [0, 0.5], act: 'relu' },
          { W: [[1, 2]], b: [0], act: 'linear' },
        ],
      },
      data: { X: [[1]], Y: [[1]] },
      opt: defaultOptimizer('sgd', 0.02),
    }),
  'gd-step': () =>
    base({
      id: 'gd-step',
      net: makeNetwork([1, 1], ['linear'], () => 3, () => 0),
      data: { X: [[2]], Y: [[10]] },
      frozen: ['b1[0]'],
      opt: defaultOptimizer('sgd', 0.05),
    }),
  'lr-compare': () =>
    base({
      id: 'lr-compare',
      net: makeNetwork([1, 1], ['linear'], () => 3, () => 0),
      data: { X: [[2]], Y: [[10]] },
      frozen: ['b1[0]'],
      opt: defaultOptimizer('sgd', 0.1),
    }),
  'mini-batch': () => {
    const xs = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2];
    const noise = [0.1, -0.2, 0.15, -0.05, 0.1, -0.1, 0.2, -0.15];
    return base({
      id: 'mini-batch',
      net: makeNetwork([1, 1], ['linear'], () => 0, () => 0),
      data: { X: xs.map((x) => [x]), Y: xs.map((x, i) => [2 * x + 1 + noise[i]]) },
      batch: { size: 2, shuffle: true, seed: 3 },
      opt: defaultOptimizer('sgd', 0.1),
    });
  },
  overfit: () =>
    base({
      id: 'overfit',
      net: initialize(makeNetwork([1, 8, 1], ['tanh', 'linear']), 'xavier', 4),
      data: { X: overfitX.map((x) => [x]), Y: overfitX.map((x, i) => [truth(x) + overfitNoise[i]]) },
      val: { X: valX.map((x) => [x]), Y: valX.map((x) => [truth(x) + normal(valRng, 0, 0.25)]) },
      opt: defaultOptimizer('adam', 0.03),
    }),
};

export const PRESET_IDS = Object.keys(PRESETS) as PresetId[];

export const loadPreset = (id: PresetId) => PRESETS[id]();

/** Boolean training mask (true = trainable) in flat parameter order. */
export const maskFor = (net: Network, frozen: string[]) => paramRefs(net).map((p) => !frozen.includes(p.id));

/** The true function behind the overfitting data, for plotting. */
export const overfitTruth = truth;
