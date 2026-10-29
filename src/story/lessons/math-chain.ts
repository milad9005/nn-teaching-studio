// Lesson "chain" math: the two-neuron chain (preset "chain-two": x = 2, w₁ = 3, w₂ = 2, biases 0,
// ReLU, ReLU, target 10 → ŷ = 12, L = 4). Forward values, the three local rates along the path of
// w₁, the engine's backprop gradients, and a real nudge. Plain names only (no matrix indices).

import { actDeriv, lossSample, predict, sampleResult, setParam, type Dataset, type LossName, type LossOptions, type Network } from '../../engine';

export interface Chain {
  x: number;
  y: number;
  w1: number;
  w2: number;
  b1: number;
  b2: number;
  /** What the first neuron outputs. */
  a1: number;
  /** What the network outputs. */
  yhat: number;
  L: number;
  /** Last link: how much the loss moves per nudge of the output, 2(ŷ − y). */
  rLast: number;
  /** Middle link: how much the output moves per nudge of a₁, w₂ · ReLU slope. */
  rMid: number;
  /** First link: how much a₁ moves per nudge of w₁, x · ReLU slope. */
  rFirst: number;
  /** ReLU slopes of the two neurons (0 or 1). */
  relu1: number;
  relu2: number;
  /** Blame that reaches the first neuron's output: ∂L/∂a₁. */
  dA1: number;
  /** The engine's backprop answers. */
  gw1: number;
  gw2: number;
}

export function chainOf(net: Network, data: Dataset, loss: LossName = 'mse', lossOpts: LossOptions = {}): Chain {
  const x = data.X[0][0];
  const y = data.Y[0][0];
  const one = sampleResult(net, [x], [y], loss, lossOpts);
  const z1 = one.trace.z[0][0];
  const z2 = one.trace.z[1][0];
  const relu1 = actDeriv(net.layers[0].act, z1);
  const relu2 = actDeriv(net.layers[1].act, z2);
  const w1 = net.layers[0].W[0][0];
  const w2 = net.layers[1].W[0][0];
  return {
    x,
    y,
    w1,
    w2,
    b1: net.layers[0].b[0],
    b2: net.layers[1].b[0],
    a1: one.trace.a[1][0],
    yhat: one.trace.a[2][0],
    L: one.loss,
    rLast: one.dOut[0],
    rMid: w2 * relu2,
    rFirst: x * relu1,
    relu1,
    relu2,
    dA1: one.back.da[1][0],
    gw1: one.back.grads.W[0][0][0],
    gw2: one.back.grads.W[1][0][0],
  };
}

export type Knob = 'w1' | 'w2';
const REF = {
  w1: { kind: 'W' as const, layer: 0, j: 0, i: 0, id: 'W1[0,0]' },
  w2: { kind: 'W' as const, layer: 1, j: 0, i: 0, id: 'W2[0,0]' },
};

/** Really turn one knob by `h` and run the network again: the loss before and after. */
export function nudgeChain(net: Network, data: Dataset, which: Knob, h: number, loss: LossName = 'mse', lossOpts: LossOptions = {}) {
  const c = chainOf(net, data, loss, lossOpts);
  const from = which === 'w1' ? c.w1 : c.w2;
  const moved = setParam(net, REF[which], from + h);
  const after = lossSample(loss, predict(moved, [c.x]), [c.y], lossOpts);
  return { from, to: from + h, before: c.L, after, change: after - c.L };
}
