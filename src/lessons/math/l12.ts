// Lesson 12 math: gradient descent on L(w) = (x·w + b − y)² (one weight, one example, MSE).
// The engine does the actual updates; these closed forms explain and predict them.

/** Second derivative L''(w) = 2x²: how fast the slope changes. */
export const curvature = (x: number) => 2 * x * x;

/** The weight with zero loss. */
export const wStar = (x: number, y: number, b: number) => (Math.abs(x) > 1e-12 ? (y - b) / x : NaN);

/**
 * One SGD step maps the distance to the minimum by a constant factor:
 * w_new − w* = (1 − η·L'') (w_old − w*).
 */
export const gdFactor = (lr: number, x: number) => 1 - lr * curvature(x);

/** Closed-form weight after k steps of SGD from w0. */
export const closedFormW = (w0: number, k: number, lr: number, x: number, y: number, b: number) => {
  const ws = wStar(x, y, b);
  return ws + gdFactor(lr, x) ** k * (w0 - ws);
};

/** The one update rule of this lesson. */
export const sgdUpdate = (w: number, lr: number, g: number) => w - lr * g;
