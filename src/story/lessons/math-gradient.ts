// Lesson "gradient": the loss over two knobs, L(w, b) = (2w + b − 10)², its partial derivatives,
// gradient descent on both knobs, and its height lines (they are straight: 2w + b − 10 = ±√c).

export const X = 2;
export const Y = 10;
export const START: [number, number] = [3, 0];

export const err = (w: number, b: number) => X * w + b - Y;
export const loss2 = (w: number, b: number) => err(w, b) ** 2;
/** ∂L/∂w = 2(2w + b − 10) · 2 */
export const dLdw = (w: number, b: number) => 2 * err(w, b) * X;
/** ∂L/∂b = 2(2w + b − 10) · 1 */
export const dLdb = (w: number, b: number) => 2 * err(w, b);
export const grad = (w: number, b: number): [number, number] => [dLdw(w, b), dLdb(w, b)];

/** Gradient descent on both knobs: each steps against its own slope. */
export function path2(p0: [number, number], eta: number, n: number): [number, number][] {
  const out: [number, number][] = [p0];
  for (let i = 0; i < n; i++) {
    const [w, b] = out[i];
    const [gw, gb] = grad(w, b);
    out.push([w - eta * gw, b - eta * gb]);
  }
  return out;
}

export type Pt = [number, number];
export interface Box {
  w0: number;
  w1: number;
  b0: number;
  b1: number;
}

/**
 * The height line L = c inside the box. Because L only depends on e = 2w + b − 10, every height line
 * is a straight line 2w + b = 10 + e with e = ±√c. Returns the segment for one signed e.
 */
export function levelSegment(e: number, box: Box): [Pt, Pt] | null {
  const k = Y + e; // 2w + b = k
  const pts: Pt[] = [];
  const add = (p: Pt) => {
    if (p[0] >= box.w0 - 1e-9 && p[0] <= box.w1 + 1e-9 && p[1] >= box.b0 - 1e-9 && p[1] <= box.b1 + 1e-9 && !pts.some((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) < 1e-9)) pts.push(p);
  };
  add([box.w0, k - X * box.w0]);
  add([box.w1, k - X * box.w1]);
  add([(k - box.b0) / X, box.b0]);
  add([(k - box.b1) / X, box.b1]);
  return pts.length >= 2 ? [pts[0], pts[1]] : null;
}
