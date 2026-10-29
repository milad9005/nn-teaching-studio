// Optimizers on flat parameter vectors, so the same code drives networks and toy objectives.
//   SGD:       theta <- theta - lr * g
//   Momentum:  v <- beta * v + g;              theta <- theta - lr * v          (PyTorch convention)
//   Adam:      m <- b1 m + (1-b1) g;  s <- b2 s + (1-b2) g^2;
//              theta <- theta - lr * mhat / (sqrt(shat) + eps),  mhat = m/(1-b1^t), shat = s/(1-b2^t)

export type OptimizerName = 'sgd' | 'momentum' | 'adam';

export interface OptimizerConfig {
  name: OptimizerName;
  lr: number;
  momentum: number;
  beta1: number;
  beta2: number;
  eps: number;
  /** Rescale the gradient when its L2 norm exceeds this. null = no clipping. */
  clipNorm: number | null;
}

export const defaultOptimizer = (name: OptimizerName = 'sgd', lr = 0.1): OptimizerConfig => ({
  name,
  lr,
  momentum: 0.9,
  beta1: 0.9,
  beta2: 0.999,
  eps: 1e-8,
  clipNorm: null,
});

export interface OptimizerState {
  t: number;
  v: number[];
  m: number[];
  s: number[];
}

export const initOptState = (n: number): OptimizerState => ({
  t: 0,
  v: new Array(n).fill(0),
  m: new Array(n).fill(0),
  s: new Array(n).fill(0),
});

export type StepResult =
  | { ok: true; theta: number[]; state: OptimizerState; delta: number[]; clipped: boolean }
  | { ok: false; error: 'nonFiniteGrad' | 'nonFiniteParam' };

export const norm = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

/**
 * One optimizer step. `mask[k] = false` freezes parameter k (its gradient is treated as 0
 * and it never moves). Returns an error instead of producing NaN parameters.
 */
export function optimizerStep(
  theta: number[],
  grad: number[],
  cfg: OptimizerConfig,
  state: OptimizerState,
  mask?: boolean[],
): StepResult {
  if (grad.some((g) => !Number.isFinite(g))) return { ok: false, error: 'nonFiniteGrad' };
  let g = mask ? grad.map((x, k) => (mask[k] ? x : 0)) : grad.slice();
  let clipped = false;
  if (cfg.clipNorm !== null) {
    const n = norm(g);
    if (n > cfg.clipNorm) {
      g = g.map((x) => (x * cfg.clipNorm!) / n);
      clipped = true;
    }
  }
  const t = state.t + 1;
  const v = state.v.slice();
  const m = state.m.slice();
  const s = state.s.slice();
  const delta = g.map((gk, k) => {
    if (mask && !mask[k]) return 0;
    switch (cfg.name) {
      case 'sgd':
        return -cfg.lr * gk;
      case 'momentum':
        v[k] = cfg.momentum * v[k] + gk;
        return -cfg.lr * v[k];
      case 'adam': {
        m[k] = cfg.beta1 * m[k] + (1 - cfg.beta1) * gk;
        s[k] = cfg.beta2 * s[k] + (1 - cfg.beta2) * gk * gk;
        const mh = m[k] / (1 - cfg.beta1 ** t);
        const sh = s[k] / (1 - cfg.beta2 ** t);
        return (-cfg.lr * mh) / (Math.sqrt(sh) + cfg.eps);
      }
    }
  });
  const next = theta.map((x, k) => x + delta[k]);
  if (next.some((x) => !Number.isFinite(x))) return { ok: false, error: 'nonFiniteParam' };
  return { ok: true, theta: next, state: { t, v, m, s }, delta, clipped };
}

/** Run an optimizer on a scalar objective with a known gradient; returns the visited points. */
export function optimizePath(
  f: (p: number[]) => number,
  grad: (p: number[]) => number[],
  start: number[],
  cfg: OptimizerConfig,
  steps: number,
  divergeAt = 1e6,
): { points: number[][]; losses: number[]; diverged: boolean } {
  let theta = start.slice();
  let st = initOptState(theta.length);
  const points = [theta.slice()];
  const losses = [f(theta)];
  for (let i = 0; i < steps; i++) {
    const r = optimizerStep(theta, grad(theta), cfg, st);
    if (!r.ok) return { points, losses, diverged: true };
    theta = r.theta;
    st = r.state;
    const L = f(theta);
    points.push(theta.slice());
    losses.push(L);
    if (!Number.isFinite(L) || Math.abs(L) > divergeAt || theta.some((x) => Math.abs(x) > divergeAt))
      return { points, losses, diverged: true };
  }
  return { points, losses, diverged: false };
}
