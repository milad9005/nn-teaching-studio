// Lesson 15 math: 2-D toy objectives and optimizer traces that keep the optimizer's internal
// state (velocity, moments) at every step so the lesson can show the exact update numbers.
// Steps are made by the engine's optimizerStep, the same function that trains networks.

import { initOptState, optimizerStep, type OptimizerConfig, type OptimizerState } from '../../engine';

export type ObjId = 'valley' | 'curved';

export interface ObjParams {
  /** Valley: L = 0.5 (a w1^2 + b w2^2). */
  a: number;
  b: number;
  /** Curved valley: L = (1 - w1)^2 + c (w2 - w1^2)^2. */
  c: number;
}

export interface Objective {
  id: ObjId;
  f: (p: number[]) => number;
  grad: (p: number[]) => number[];
  start: [number, number];
  min: [number, number];
  xRange: [number, number];
  yRange: [number, number];
}

export function makeObjective(id: ObjId, q: ObjParams): Objective {
  if (id === 'valley')
    return {
      id,
      f: ([x, y]) => 0.5 * (q.a * x * x + q.b * y * y),
      grad: ([x, y]) => [q.a * x, q.b * y],
      start: [-4.5, 1.5],
      min: [0, 0],
      xRange: [-5.2, 1.6],
      yRange: [-2.2, 2.2],
    };
  return {
    id,
    f: ([x, y]) => (1 - x) ** 2 + q.c * (y - x * x) ** 2,
    grad: ([x, y]) => [-2 * (1 - x) - 4 * q.c * x * (y - x * x), 2 * q.c * (y - x * x)],
    start: [-1.5, 2],
    min: [1, 1],
    xRange: [-2, 2],
    yRange: [-1, 3],
  };
}

export interface Trace {
  points: number[][];
  losses: number[];
  /** grads[k] = gradient at points[k] (used by step k -> k+1). */
  grads: number[][];
  /** states[k] = optimizer state after k steps (states[0] is all zeros). */
  states: OptimizerState[];
  diverged: boolean;
}

export function tracePath(obj: Objective, cfg: OptimizerConfig, steps: number, divergeAt = 1e6): Trace {
  let theta = obj.start.slice();
  let st = initOptState(2);
  const points = [theta.slice()];
  const losses = [obj.f(theta)];
  const grads: number[][] = [];
  const states = [st];
  for (let i = 0; i < steps; i++) {
    const g = obj.grad(theta);
    const r = optimizerStep(theta, g, cfg, st);
    if (!r.ok) return { points, losses, grads, states, diverged: true };
    grads.push(g);
    theta = r.theta;
    st = r.state;
    points.push(theta.slice());
    states.push(st);
    const L = obj.f(theta);
    losses.push(L);
    if (!Number.isFinite(L) || L > divergeAt || theta.some((v) => Math.abs(v) > divergeAt)) return { points, losses, grads, states, diverged: true };
  }
  return { points, losses, grads, states, diverged: false };
}

/** Grid of log(1 + L) values for contour plots (log spacing shows the valley floor). */
export function contourGrid(obj: Objective, n = 70) {
  const xs = Array.from({ length: n }, (_, i) => obj.xRange[0] + ((obj.xRange[1] - obj.xRange[0]) * i) / (n - 1));
  const ys = Array.from({ length: n }, (_, i) => obj.yRange[0] + ((obj.yRange[1] - obj.yRange[0]) * i) / (n - 1));
  const z = ys.map((y) => xs.map((x) => Math.log1p(obj.f([x, y]))));
  return { xs, ys, z };
}
