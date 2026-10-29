// Loss functions. Normalization is the same everywhere:
//   per-sample loss  l(yhat, y) = mean over output units (except CCE, which sums over classes)
//   batch loss       L = mean of per-sample losses over the samples in the batch
// lossGrad returns d l / d yhat for one sample; network.ts divides by the batch size.

export type LossName = 'mse' | 'mae' | 'huber' | 'bce' | 'cce';

export interface LossOptions {
  /** Huber threshold. */
  delta?: number;
}

/** Probabilities are clipped to [EPS, 1-EPS] so log() never sees 0. */
export const EPS = 1e-12;
const clipP = (p: number) => Math.min(1 - EPS, Math.max(EPS, p));

function huber(r: number, d: number) {
  const a = Math.abs(r);
  return a <= d ? 0.5 * r * r : d * (a - 0.5 * d);
}

export function lossSample(name: LossName, yhat: number[], y: number[], o: LossOptions = {}): number {
  const K = yhat.length;
  let s = 0;
  switch (name) {
    case 'mse':
      for (let k = 0; k < K; k++) s += (yhat[k] - y[k]) ** 2;
      return s / K;
    case 'mae':
      for (let k = 0; k < K; k++) s += Math.abs(yhat[k] - y[k]);
      return s / K;
    case 'huber':
      for (let k = 0; k < K; k++) s += huber(yhat[k] - y[k], o.delta ?? 1);
      return s / K;
    case 'bce':
      for (let k = 0; k < K; k++) {
        const p = clipP(yhat[k]);
        s += -(y[k] * Math.log(p) + (1 - y[k]) * Math.log(1 - p));
      }
      return s / K;
    case 'cce':
      for (let k = 0; k < K; k++) s += -y[k] * Math.log(clipP(yhat[k]));
      return s;
  }
}

/** d(per-sample loss) / d yhat. MAE uses sign(0) = 0 at the kink. */
export function lossGrad(name: LossName, yhat: number[], y: number[], o: LossOptions = {}): number[] {
  const K = yhat.length;
  switch (name) {
    case 'mse':
      return yhat.map((v, k) => (2 * (v - y[k])) / K);
    case 'mae':
      return yhat.map((v, k) => Math.sign(v - y[k]) / K);
    case 'huber': {
      const d = o.delta ?? 1;
      return yhat.map((v, k) => {
        const r = v - y[k];
        return (Math.abs(r) <= d ? r : d * Math.sign(r)) / K;
      });
    }
    case 'bce':
      return yhat.map((v, k) => {
        const p = clipP(v);
        return (-y[k] / p + (1 - y[k]) / (1 - p)) / K;
      });
    case 'cce':
      return yhat.map((v, k) => -y[k] / clipP(v));
  }
}

/** Which output activation each loss expects, for validation and UI hints. */
export const LOSS_OUTPUT: Record<LossName, 'any' | 'sigmoid' | 'softmax'> = {
  mse: 'any',
  mae: 'any',
  huber: 'any',
  bce: 'sigmoid',
  cce: 'softmax',
};
