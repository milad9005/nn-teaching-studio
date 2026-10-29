// Lesson "neuron" math: one neuron ŷ = w·x + b (and with two inputs, w₁x₁ + w₂x₂ + b).
// Every value comes from the engine (a one-layer network + predict).

import { predict, type Network } from '../../engine';

/** The running example: 2 notebooks, the knob w = 3 (price of one notebook), no fee. */
export const NOTEBOOKS = 2;
export const W0 = 3;
export const B0 = 0;
/** The second input of step 7: 1 pen at 2 coins. */
export const PENS = 1;
export const W_PEN = 2;

/** Step 2: the knob turns 3 → 5 → 1 → 3. Step 4: the fee turns 0 → 2 → 0. */
export const W_SWEEP = [3, 5, 1, 3];
export const B_SWEEP = [0, 2, 0];

/** A plain (no bend) neuron with these input weights, as an engine network. */
export const neuronNet = (ws: number[], b: number): Network => ({
  inputSize: ws.length,
  layers: [{ W: [ws.slice()], b: [b], act: 'linear' }],
});

/** ŷ = w₁x₁ + … + b, computed by the engine. */
export const neuronOut = (ws: number[], b: number, xs: number[]) => predict(neuronNet(ws, b), xs)[0];

/** One input: ŷ = w·x + b. */
export const guess = (w: number, b: number, x: number) => neuronOut([w], b, [x]);

/** Each input's share w_i · x_i, in input order. */
export const shares = (ws: number[], xs: number[]) => ws.map((w, i) => w * xs[i]);

/**
 * The step triangle on the line at x0: one step right, then up by the slope.
 * `rise` is measured on the neuron itself (ŷ at x0 + 1 minus ŷ at x0), so it equals w.
 */
export function stepTriangle(w: number, b: number, x0: number) {
  const y0 = guess(w, b, x0);
  const y1 = guess(w, b, x0 + 1);
  return { start: [x0, y0] as [number, number], corner: [x0 + 1, y0] as [number, number], end: [x0 + 1, y1] as [number, number], run: 1, rise: y1 - y0 };
}
