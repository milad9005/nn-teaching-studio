import { describe, expect, it } from 'vitest';
import { loss, path, slope, step, stepsToReach, W_BEST } from './math-descent';

describe('descent: L(w) = (2w − 10)²', () => {
  it('starts at w = 3 with loss 16 and slope −16', () => {
    expect(loss(3)).toBe(16);
    expect(slope(3)).toBe(-16);
  });
  it('one step with η = 0.05: w 3 → 3.8, L 16 → 5.76', () => {
    expect(step(3, 0.05)).toBeCloseTo(3.8, 12);
    expect(loss(step(3, 0.05))).toBeCloseTo(5.76, 12);
  });
  it('converges to w = 5 with η = 0.05', () => {
    expect(W_BEST).toBe(5);
    expect(path(3, 0.05, 40).at(-1)).toBeCloseTo(5, 6);
    expect(slope(5)).toBe(0);
  });
  it('η = 0.005 crawls, η = 0.27 flies off', () => {
    expect(stepsToReach(3, 0.005, 0.01)).toBeGreaterThan(100);
    expect(stepsToReach(3, 0.05, 0.01)).toBeLessThan(15);
    const p = path(3, 0.27, 8);
    for (let i = 1; i < p.length; i++) expect(Math.abs(p[i] - 5)).toBeGreaterThan(Math.abs(p[i - 1] - 5));
    expect(Math.sign(p[1] - 5)).toBe(-Math.sign(p[2] - 5));
  });
});
