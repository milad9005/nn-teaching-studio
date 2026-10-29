// Lesson "network" math: neurons in a row, the ReLU bend, bent lines added up, and the forward
// pass of preset "chain-two". Every network value comes from the engine (makeNetwork + predict).

import { act, forward, loadPreset, makeNetwork, predict, type Network } from '../../engine';

export type Bend = 'linear' | 'relu';

/** Two one-input neurons in a row, each followed by the same bend (or none). */
export const chainNet = (w1: number, b1: number, w2: number, b2: number, bend: Bend): Network =>
  makeNetwork([1, 1, 1], [bend, bend], (l) => (l === 0 ? w1 : w2), (l) => (l === 0 ? b1 : b2));

export const chainOut = (w1: number, b1: number, w2: number, b2: number, bend: Bend, x: number) => predict(chainNet(w1, b1, w2, b2, bend), [x])[0];

/** The values along the chain for input x: [x, after neuron 1, after neuron 2]. */
export function chainValues(w1: number, b1: number, w2: number, b2: number, bend: Bend, x: number): number[] {
  const t = forward(chainNet(w1, b1, w2, b2, bend), [x]);
  return [t.a[0][0], t.a[1][0], t.a[2][0]];
}

/** The one straight neuron that two straight neurons in a row really are. */
export const collapsed = (w1: number, b1: number, w2: number, b2: number) => ({ w: w2 * w1, b: w2 * b1 + b2 });

export const relu = (v: number) => act('relu', v);

/** A one-neuron network with a bend: ReLU(w x + b). */
export const bentNeuron = (w: number, b: number, x: number) => predict(makeNetwork([1, 1], ['relu'], () => w, () => b), [x])[0];

/** Where w x + b = 0, the corner of ReLU(w x + b); null when w = 0. */
export const bendAt = (w: number, b: number): number | null => (Math.abs(w) > 1e-12 ? -b / w + 0 : null);

export interface Hinge {
  /** Weight into the hidden neuron. */
  w: number;
  /** Bias of the hidden neuron. */
  b: number;
  /** Weight from the hidden neuron into the output neuron. */
  v: number;
}

/**
 * Four bent lines, added in this order. Together they pass exactly through the U-shaped curve
 * x · x at x = −2, −1, 0, 1, 2.
 */
export const HINGES: Hinge[] = [
  { w: 1, b: 0, v: 1 },
  { w: -1, b: 0, v: 1 },
  { w: 1, b: -1, v: 2 },
  { w: -1, b: -1, v: 2 },
];

/** 1 input → hidden ReLU neurons → 1 plain output neuron that adds them up. */
export const hingeNet = (units: Hinge[]): Network =>
  makeNetwork(
    [1, units.length, 1],
    ['relu', 'linear'],
    (l, j, i) => (l === 0 ? units[j].w : units[i].v),
    (l, j) => (l === 0 ? units[j].b : 0),
  );

/** One hidden neuron's share of the output at x: v · ReLU(w x + b). */
export const piece = (u: Hinge, x: number) => u.v * relu(u.w * x + u.b);

/** The network's output with the first n hinges (n = 0 gives the flat line 0). */
export const hingeSum = (n: number, x: number) => (n <= 0 ? 0 : predict(hingeNet(HINGES.slice(0, n)), [x])[0]);

/** The forward pass of preset chain-two, read from the preset: x, z₁, a₁, z₂, ŷ and the target. */
export function chainTwo() {
  const p = loadPreset('chain-two');
  const x = p.data.X[0][0];
  const t = forward(p.net, [x]);
  return {
    x,
    w1: p.net.layers[0].W[0][0],
    b1: p.net.layers[0].b[0],
    w2: p.net.layers[1].W[0][0],
    b2: p.net.layers[1].b[0],
    z1: t.z[0][0],
    a1: t.a[1][0],
    z2: t.z[1][0],
    yhat: t.a[2][0],
    target: p.data.Y[0][0],
  };
}
