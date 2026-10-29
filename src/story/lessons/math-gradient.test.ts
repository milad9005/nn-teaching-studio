import { describe, expect, it } from 'vitest';
import { dLdb, dLdw, grad, levelSegment, loss2, path2 } from './math-gradient';

describe('gradient: L(w, b) = (2w + b − 10)²', () => {
  it('at (3, 0): L = 16, gradient (−16, −8)', () => {
    expect(loss2(3, 0)).toBe(16);
    expect(grad(3, 0)).toEqual([-16, -8]);
  });
  it('partials match a tiny nudge', () => {
    const h = 1e-6;
    expect((loss2(3 + h, 0) - loss2(3, 0)) / h).toBeCloseTo(dLdw(3, 0), 4);
    expect((loss2(3, h) - loss2(3, 0)) / h).toBeCloseTo(dLdb(3, 0), 4);
  });
  it('descent on both knobs reaches the valley floor 2w + b = 10', () => {
    const p = path2([3, 0], 0.05, 30);
    expect(p[1][0]).toBeCloseTo(3.8, 12);
    expect(p[1][1]).toBeCloseTo(0.4, 12);
    const [w, b] = p.at(-1)!;
    expect(loss2(w, b)).toBeLessThan(1e-12);
    expect(w).toBeCloseTo(4.6, 6);
    expect(b).toBeCloseTo(0.8, 6);
  });
  it('height lines are clipped to the box and lie at the right height', () => {
    const box = { w0: 0, w1: 9, b0: -3, b1: 4 };
    const s = levelSegment(-4, box)!;
    for (const [w, b] of s) expect(loss2(w, b)).toBeCloseTo(16, 9);
    expect(levelSegment(100, box)).toBeNull();
  });
});
