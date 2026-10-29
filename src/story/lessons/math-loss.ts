// Lesson "loss": how wrong is the machine? The running example (2 notebooks, real price 10, the
// one-neuron machine with w = 3), the error ŷ − y, its square, and the average over 4 receipts.
// Every number the scene shows comes from here.

import { lossSample, predict, type Network } from '../../engine';

/** The running example: 2 notebooks, the knob w = 3, the real price 10. */
export const X = 2;
export const W0 = 3;
export const Y = 10;

/** The one-neuron price machine ŷ = w·x (no fee), as an engine network. */
const neuron = (w: number): Network => ({ inputSize: 1, layers: [{ W: [[w]], b: [0], act: 'linear' }] });

/** The machine's guess ŷ for x notebooks. */
export const guess = (w: number, x: number) => predict(neuron(w), [x])[0];

/** The error: guess minus answer, ŷ − y. Negative means too few. */
export const errorOf = (yhat: number, y: number) => yhat - y;

/** One example's loss (ŷ − y)², from the engine's squared error. */
export const lossOf = (yhat: number, y: number) => lossSample('mse', [yhat], [y]);

/** Step 3: a guess just as wrong as 6, but on the other side of 10. */
export const mirrorGuess = (yhat: number, y: number) => y - errorOf(yhat, y);

/** Step 4: the error sizes compared, smallest first. Each becomes a square anchored at y. */
export const COMPARE = [1, 4, 8];

/**
 * Step 6: four receipts from the shop (notebooks bought → coins really paid). A notebook costs
 * about 5, with small differences. The second one is the running example.
 */
export const RECEIPTS: { x: number; y: number }[] = [
  { x: 1, y: 6 },
  { x: 2, y: 10 },
  { x: 3, y: 15 },
  { x: 4, y: 19 },
];

export interface ReceiptRow {
  x: number;
  y: number;
  yhat: number;
  err: number;
  loss: number;
}

/** Every receipt's guess, error and loss for the knob w. */
export const receiptRows = (w: number): ReceiptRow[] =>
  RECEIPTS.map(({ x, y }) => {
    const yhat = guess(w, x);
    return { x, y, yhat, err: errorOf(yhat, y), loss: lossOf(yhat, y) };
  });

/** The average of the receipts' losses: one number for the whole job. */
export const meanLoss = (w: number) => {
  const rows = receiptRows(w);
  return rows.reduce((s, r) => s + r.loss, 0) / rows.length;
};

/**
 * Step 6 animation: a square of area `area` turns into a column of width `cw` with the same
 * area (morph 0 → 1), then every column levels out to the height of the average (level 0 → 1).
 * Returns the shape's width and height in the same units as the square's side.
 */
export function pourShape(area: number, cw: number, mean: number, morph: number, level: number) {
  const side = Math.sqrt(area);
  const width = side + (cw - side) * morph;
  const own = width > 1e-9 ? area / width : 0;
  const height = own + (mean / cw - own) * level * morph;
  return { width, height };
}
