import { describe, expect, it } from 'vitest';
import { bendAt, bentNeuron, chainOut, chainTwo, chainValues, collapsed, hingeSum, HINGES, piece, relu } from './math-network';

describe('network lesson math', () => {
  it('2 → ×3 → 6 → ×2 → 12', () => {
    expect(chainValues(3, 0, 2, 0, 'linear', 2)).toEqual([2, 6, 12]);
  });
  it('two straight neurons in a row are one straight neuron ×6', () => {
    const c = collapsed(3, 0, 2, 0);
    expect(c).toEqual({ w: 6, b: 0 });
    for (const x of [-2, -0.5, 0, 1.3, 3]) expect(chainOut(3, 0, 2, 0, 'linear', x)).toBeCloseTo(c.w * x + c.b);
  });
  it('ReLU: negatives become 0, positives stay', () => {
    expect(relu(-2)).toBe(0);
    expect(relu(1.5)).toBe(1.5);
  });
  it('ReLU(3x − 2) bends at x = 2/3', () => {
    expect(bendAt(3, -2)).toBeCloseTo(2 / 3);
    expect(bentNeuron(3, -2, 0)).toBe(0);
    expect(bentNeuron(3, -2, 2)).toBe(4);
  });
  it('four bent lines add up to x·x at the integers', () => {
    for (const x of [-2, -1, 0, 1, 2]) expect(hingeSum(4, x)).toBeCloseTo(x * x);
    for (const x of [-1.7, 0.4, 2.2]) expect(hingeSum(4, x)).toBeCloseTo(HINGES.reduce((s, u) => s + piece(u, x), 0));
    expect(hingeSum(0, 1)).toBe(0);
  });
  it('preset chain-two: x = 2, 6, 6, 12, ŷ = 12, real price 10', () => {
    const c = chainTwo();
    expect([c.x, c.z1, c.a1, c.z2, c.yhat, c.target]).toEqual([2, 6, 6, 12, 12, 10]);
  });
});
