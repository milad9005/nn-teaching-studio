// Lesson "derivative": slope = derivative. The running curve L(w) = (2w − 10)² (2 notebooks,
// real price 10), measured by a tiny nudge, and the camera that zooms in at w = 3.
// Every number the scene shows comes from here.

export const X = 2;
export const Y = 10;
export const W0 = 3;

/** The loss for the knob w: L(w) = (2w − 10)². */
export const loss = (w: number) => (X * w - Y) ** 2;

/** The exact slope of the curve at w (used to draw tangents; the lesson measures it by a nudge). */
export const slope = (w: number) => 2 * (X * w - Y) * X;

/** The bottom of the U: where 2w = 10. */
export const W_BEST = Y / X;

/** Step 1: the knob turns a little, 3 → 3.5. */
export const W_TURN = 3.5;

/** Step 6: the tiny nudge. */
export const NUDGE = 0.001;

/** Nudge w by h and measure: the change in loss, and the change in loss ÷ nudge. */
export function nudge(w: number, h = NUDGE) {
  const L0 = loss(w);
  const w1 = w + h;
  const L1 = loss(w1);
  return { w0: w, w1, L0, L1, dw: h, dL: L1 - L0, rate: (L1 - L0) / h };
}

/** Step 4: the points where we look at the steepness. */
export const TANGENT_AT = [1, 3, 5, 7];

/** Step 8: the auto-demo of the sign: negative, positive, zero. */
export const SIGN_DEMO = [3, 7, 5];

/** Lesson 2's line: the price of x notebooks at w = 3, and its step triangle "+1 → +3". */
export const LINE_W = 3;
export const line = (x: number) => LINE_W * x;
export const LINE_TRIANGLES = [0.4, 1.6, 2.8];
export const stepTriangle = (x0: number) => ({ x0, y0: line(x0), x1: x0 + 1, y1: line(x0 + 1), rise: line(x0 + 1) - line(x0) });

// ---- The camera ---------------------------------------------------------------------------

export interface View {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

/** The whole U curve. */
export const BASE: View = { x0: -0.5, x1: 10.5, y0: -10, y1: 108 };
/** Lesson 2's line, for the recall. */
export const LINE_VIEW: View = { x0: -0.3, x1: 4.3, y0: -1.2, y1: 14 };

/** Zoom levels of step 5 (×1, ×10, ×100) and of the nudge in step 6 (×2000), as powers of 10. */
export const Z_STEPS = [0, 1, 2];
export const Z_NUDGE = Math.log10(2000);

const smooth = (t: number) => {
  const u = Math.max(0, Math.min(1, t));
  return u * u * (3 - 2 * u);
};

/**
 * The camera for zoom level z (magnification 10^z). It zooms in on the point (3, L(3)); by ×10 the
 * point is centred, and between ×100 and ×2000 it slides to the nudge triangle.
 */
export function zoomView(z: number): View {
  const fx = W0;
  const fy = loss(W0);
  const bx = (BASE.x0 + BASE.x1) / 2;
  const by = (BASE.y0 + BASE.y1) / 2;
  const hx = ((BASE.x1 - BASE.x0) / 2) * 10 ** -z;
  const hy = ((BASE.y1 - BASE.y0) / 2) * 10 ** -z;
  const away = 10 ** -z * Math.max(0, 1 - z);
  const n = nudge(W0);
  const toMid = smooth((z - Z_STEPS[2]) / (Z_NUDGE - Z_STEPS[2]));
  // At ×2000 the triangle sits a little left of and below the middle, leaving the top right free.
  const cx = fx + (bx - fx) * away + ((n.w0 + n.w1) / 2 + 0.24 * hx - fx) * toMid;
  const cy = fy + (by - fy) * away + ((n.L0 + n.L1) / 2 + 0.16 * hy - fy) * toMid;
  return { x0: cx - hx, x1: cx + hx, y0: cy - hy, y1: cy + hy };
}

/** Blend two views (0 → a, 1 → b). */
export const mixView = (a: View, b: View, t: number): View => ({
  x0: a.x0 + (b.x0 - a.x0) * t,
  x1: a.x1 + (b.x1 - a.x1) * t,
  y0: a.y0 + (b.y0 - a.y0) * t,
  y1: a.y1 + (b.y1 - a.y1) * t,
});

/** How many decimals a tick step needs (0.25 → 2, 0.001 → 3, 5 → 0). */
export function stepDigits(step: number) {
  for (let d = 0; d < 10; d++) {
    const v = step * 10 ** d;
    if (Math.abs(v - Math.round(v)) < 1e-6 * Math.max(1, v)) return d;
  }
  return 10;
}
