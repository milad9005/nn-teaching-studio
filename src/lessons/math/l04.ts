// Lesson 04 math: a 1 -> n (ReLU) -> 1 (linear) network written as a sum of hinge pieces.
//   yhat(x) = c + sum_k v_k * ReLU(w_k x + b_k)
// The network itself always comes from the engine (makeNetwork + predict); the per-neuron
// contributions are the same sum written out term by term.

import { act, loadPreset, makeNetwork, type Network } from '../../engine';

export interface Unit {
  /** Input weight w_k. */
  w: number;
  /** Hidden bias b_k. */
  b: number;
  /** Output weight v_k. */
  v: number;
}

export const MAX_UNITS = 10;

/** Engine network for the first units.length hidden neurons and output bias c. */
export function buildNet(units: Unit[], c: number): Network {
  return makeNetwork(
    [1, units.length, 1],
    ['relu', 'linear'],
    (l, j, i) => (l === 0 ? units[j].w : units[i].v),
    (l, j) => (l === 0 ? units[j].b : c),
  );
}

/** One neuron's piece of the output: v * ReLU(w x + b). */
export const contribution = (u: Unit, x: number) => u.v * act('relu', u.w * x + u.b);

/** Where w x + b = 0 (the hinge), or null when w = 0 (no hinge: the neuron is always on or always off). */
export const breakpoint = (u: Unit): number | null => (Math.abs(u.w) > 1e-12 ? -u.b / u.w + 0 : null); // "+ 0" turns -0 into 0

/** Which inputs switch the neuron on (w x + b > 0). */
export function activeSide(u: Unit): 'right' | 'left' | 'always' | 'never' {
  if (u.w > 1e-12) return 'right';
  if (u.w < -1e-12) return 'left';
  return u.b > 0 ? 'always' : 'never';
}

/**
 * Bends of the combined output inside the open interval (lo, hi). At a breakpoint the slope of
 * v ReLU(w x + b) jumps by v |w|; neurons sharing a breakpoint add their jumps, and a bend only
 * exists where the total jump is nonzero.
 */
export function kinks(units: Unit[], lo: number, hi: number, tol = 1e-9): { x: number; jump: number }[] {
  const groups: { x: number; jump: number }[] = [];
  for (const u of units) {
    const bp = breakpoint(u);
    if (bp === null || bp <= lo || bp >= hi) continue;
    const jump = u.v * Math.abs(u.w);
    const g = groups.find((q) => Math.abs(q.x - bp) < 1e-9);
    if (g) g.jump += jump;
    else groups.push({ x: bp, jump });
  }
  return groups.filter((g) => Math.abs(g.jump) > tol).sort((a, b) => a.x - b.x);
}

/** The starting neurons of preset 'one-vs-many' (target y = 0.5 x^2), read from the engine preset. */
export function presetUnits(): { units: Unit[]; c: number } {
  const net = loadPreset('one-vs-many').net;
  const [h, o] = net.layers;
  return { units: h.b.map((b, j) => ({ w: h.W[j][0], b, v: o.W[0][j] })), c: o.b[0] };
}

const u = (w: number, b: number, v: number): Unit => ({ w, b, v });

/** Neurons with hinges at 0 and ±1: the best 4-neuron fit through x = 0, ±1, ±2 of 0.5 x^2. */
const FIT4 = [u(1, 0, 0.5), u(-1, 0, 0.5), u(1, -1, 1), u(-1, -1, 1)];

/** Teaching scenarios. All but 'fit8' show more neurons that do NOT change the function. */
export const SCENARIOS = {
  preset: () => presetUnits(),
  fit4: () => ({ units: FIT4.map((q) => ({ ...q })), c: 0 }),
  /** Hinges at 0, ±0.5, ±1, ±1.5: passes exactly through all nine targets 0.5 x^2, x = -2..2. */
  fit8: () => ({
    units: [u(1, 0, 0.25), u(-1, 0, 0.25), u(1, -0.5, 0.5), u(-1, -0.5, 0.5), u(1, -1, 0.5), u(-1, -1, 0.5), u(1, -1.5, 0.5), u(-1, -1.5, 0.5)],
    c: 0,
  }),
  /** Every FIT4 neuron twice with half the output weight: 8 neurons, same function. */
  duplicate: () => ({ units: FIT4.flatMap((q) => [{ ...q, v: q.v / 2 }, { ...q, v: q.v / 2 }]), c: 0 }),
  /** FIT4 plus neurons whose hinge is at |x| = 4, outside the plotted range, and off inside it. */
  offscreen: () => ({ units: [...FIT4.map((q) => ({ ...q })), u(1, -4, 0.8), u(-1, -4, 0.8)], c: 0 }),
  /** FIT4 plus neurons with output weight 0: they compute something, but nobody listens. */
  zeroV: () => ({ units: [...FIT4.map((q) => ({ ...q })), u(1, -0.5, 0), u(-1, -0.5, 0), u(1, -1.5, 0), u(-1, -1.5, 0)], c: 0 }),
} satisfies Record<string, () => { units: Unit[]; c: number }>;

export type ScenarioId = keyof typeof SCENARIOS;

/** Neurons used to fill the unused slots up to MAX_UNITS when a scenario has fewer. */
export const SPARE: Unit[] = [u(1, -0.5, 0.25), u(-1, -0.5, 0.25), u(1, -1.5, 0.25), u(-1, -1.5, 0.25), u(0.5, 1, 0.3), u(-2, 1, -0.2), u(1, 2, 0.2), u(-1, 2, 0.2), u(2, -3, 0.1), u(-2, -3, 0.1)];

/** A full 10-slot unit list for a scenario, plus how many of them are in use. */
export function scenario(id: ScenarioId): { units: Unit[]; n: number; c: number } {
  const s = SCENARIOS[id]();
  const units = [...s.units, ...SPARE.slice(s.units.length, MAX_UNITS)].slice(0, MAX_UNITS);
  return { units, n: s.units.length, c: s.c };
}
