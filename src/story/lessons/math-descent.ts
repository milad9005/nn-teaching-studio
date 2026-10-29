// Lesson "descent": the running example L(w) = (2w − 10)² (2 notebooks, real price 10) and the
// update rule w ← w − η · dL/dw. Every number the scene shows comes from here.

export const X = 2;
export const Y = 10;
export const W0 = 3;

/** The machine's guess for 2 notebooks. */
export const guess = (w: number) => X * w;
/** The loss L(w) = (2w − 10)². */
export const loss = (w: number) => (X * w - Y) ** 2;
/** Its slope dL/dw = 2(2w − 10) · 2. */
export const slope = (w: number) => 2 * (X * w - Y) * X;
/** The best knob value, where the slope is 0. */
export const W_BEST = Y / X;

/** One step of gradient descent. */
export const step = (w: number, eta: number) => w - eta * slope(w);

/** The positions w0, w1, …, wn of n steps. */
export function path(w0: number, eta: number, n: number): number[] {
  const out = [w0];
  for (let i = 0; i < n; i++) out.push(step(out[i], eta));
  return out;
}

/** How many steps until |w − best| < tol (capped at max). */
export function stepsToReach(w0: number, eta: number, tol: number, max = 10000): number {
  let w = w0;
  for (let i = 0; i < max; i++) {
    if (Math.abs(w - W_BEST) < tol) return i;
    w = step(w, eta);
  }
  return max;
}
