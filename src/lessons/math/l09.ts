// Lesson 09 math: the loss over two weights, sampled from the engine on a grid, plus
// marching-squares contour lines of that grid (so the contour map is the real function too).

import { datasetLoss, setParam, type Dataset, type LossName, type Network, type ParamRef } from '../../engine';

export const W1_REF: ParamRef = { kind: 'W', layer: 0, j: 0, i: 0, id: 'W1[0,0]' };
export const W2_REF: ParamRef = { kind: 'W', layer: 0, j: 0, i: 1, id: 'W1[0,1]' };
export const B_REF: ParamRef = { kind: 'b', layer: 0, j: 0, i: -1, id: 'b1[0]' };

export function withWeights(net: Network, w1: number, w2: number): Network {
  return setParam(setParam(net, W1_REF, w1), W2_REF, w2);
}

/** Dataset loss at (w1, w2), everything else as in `net`. */
export const lossAt = (net: Network, data: Dataset, loss: LossName, w1: number, w2: number) => datasetLoss(withWeights(net, w1, w2), data, loss);

/** z[j][i] = L(w1s[i], w2s[j]) — rows follow w2, columns follow w1 (Plotly's convention). */
export function lossGrid(net: Network, data: Dataset, loss: LossName, w1s: number[], w2s: number[]): number[][] {
  return w2s.map((w2) => w1s.map((w1) => lossAt(net, data, loss, w1, w2)));
}

export type Pt = [number, number];

/**
 * Marching squares: line segments where the bilinear interpolation of the grid equals `level`.
 * xs index columns of z, ys index rows of z.
 */
export function contourSegments(xs: number[], ys: number[], z: number[][], level: number): [Pt, Pt][] {
  const segs: [Pt, Pt][] = [];
  const cross = (xa: number, ya: number, va: number, xb: number, yb: number, vb: number): Pt => {
    const t = (level - va) / (vb - va);
    return [xa + t * (xb - xa), ya + t * (yb - ya)];
  };
  for (let j = 0; j < ys.length - 1; j++) {
    for (let i = 0; i < xs.length - 1; i++) {
      const x0 = xs[i];
      const x1 = xs[i + 1];
      const y0 = ys[j];
      const y1 = ys[j + 1];
      const v00 = z[j][i];
      const v10 = z[j][i + 1];
      const v11 = z[j + 1][i + 1];
      const v01 = z[j + 1][i];
      const inn = (v: number) => v >= level;
      const pts: Pt[] = [];
      // Edges in order: bottom, right, top, left.
      if (inn(v00) !== inn(v10)) pts.push(cross(x0, y0, v00, x1, y0, v10));
      if (inn(v10) !== inn(v11)) pts.push(cross(x1, y0, v10, x1, y1, v11));
      if (inn(v01) !== inn(v11)) pts.push(cross(x0, y1, v01, x1, y1, v11));
      if (inn(v00) !== inn(v01)) pts.push(cross(x0, y0, v00, x0, y1, v01));
      if (pts.length === 2) segs.push([pts[0], pts[1]]);
      else if (pts.length === 4) {
        // Saddle cell: resolve with the cell-center value.
        const c = (v00 + v10 + v11 + v01) / 4;
        if (inn(c) === inn(v00)) segs.push([pts[0], pts[1]], [pts[2], pts[3]]);
        else segs.push([pts[0], pts[3]], [pts[1], pts[2]]);
      }
    }
  }
  return segs;
}

/** Contour levels that spread nicely over a loss surface with values in [0, max]. */
export function contourLevels(max: number, n = 10): number[] {
  if (!(max > 0)) return [];
  // Quadratic spacing: dense near the minimum where the bowl is flat.
  return Array.from({ length: n }, (_, k) => max * ((k + 1) / (n + 1)) ** 2);
}
