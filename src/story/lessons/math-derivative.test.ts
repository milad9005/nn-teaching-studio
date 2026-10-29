import { describe, expect, it } from 'vitest';
import { BASE, loss, nudge, slope, stepDigits, stepTriangle, W_BEST, W_TURN, Z_NUDGE, Z_STEPS, zoomView } from './math-derivative';

const inside = (v: ReturnType<typeof zoomView>, x: number, y: number) => x > v.x0 && x < v.x1 && y > v.y0 && y < v.y1;

describe('derivative lesson math', () => {
  it('the running curve: L(3) = 16, L(3.5) = 9, bottom at w = 5', () => {
    expect(loss(3)).toBe(16);
    expect(loss(W_TURN)).toBe(9);
    expect(W_BEST).toBe(5);
    expect(loss(W_BEST)).toBe(0);
  });
  it('the nudge 3 → 3.001 gives 15.984004 and a rate of about −16', () => {
    const n = nudge(3);
    expect(n.w1).toBeCloseTo(3.001, 12);
    expect(n.L1).toBeCloseTo(15.984004, 9);
    expect(n.dL).toBeCloseTo(-0.015996, 9);
    expect(n.rate).toBeCloseTo(-15.996, 6);
    expect(slope(3)).toBe(-16);
  });
  it('the sign: −16 at 3, +16 at 7, 0 at 5', () => {
    expect(slope(7)).toBe(16);
    expect(slope(5)).toBe(0);
    expect(nudge(7).rate).toBeCloseTo(16, 2);
  });
  it('lesson 2 triangle: +1 → +3 everywhere on the line', () => {
    for (const x of [0.4, 1.6, 2.8]) expect(stepTriangle(x).rise).toBeCloseTo(3, 12);
  });
  it('the camera: whole curve at ×1, w = 3 always in view, the nudge triangle in view at ×2000', () => {
    expect(zoomView(0)).toEqual(BASE);
    for (let z = 0; z <= Z_NUDGE; z += 0.05) expect(inside(zoomView(z), 3, 16)).toBe(true);
    const v = zoomView(Z_NUDGE);
    const n = nudge(3);
    expect(inside(v, n.w1, n.L1)).toBe(true);
    expect((v.x1 - v.x0) / (BASE.x1 - BASE.x0)).toBeCloseTo(1 / 2000, 9);
    const v2 = zoomView(Z_STEPS[2]);
    expect((v2.x0 + v2.x1) / 2).toBeCloseTo(3, 9);
  });
  it('up close the curve is straight: at ×100 it leaves its tangent by about 1% of the view', () => {
    const v = zoomView(2);
    const bend = Math.max(...[v.x0, v.x1].map((x) => Math.abs(loss(x) - (16 + slope(3) * (x - 3)))));
    expect(bend / (v.y1 - v.y0)).toBeLessThan(0.011);
  });
  it('tick decimals', () => {
    expect(stepDigits(5)).toBe(0);
    expect(stepDigits(0.25)).toBe(2);
    expect(stepDigits(0.001)).toBe(3);
  });
});
