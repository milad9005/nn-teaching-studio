import { describe, expect, it } from 'vitest';
import { B0, guess, neuronOut, NOTEBOOKS, PENS, shares, stepTriangle, W0, W_PEN } from './math-neuron';

describe('neuron lesson math', () => {
  it('the running example: 2 notebooks at w = 3, b = 0 guesses 6', () => {
    expect(guess(W0, B0, NOTEBOOKS)).toBe(6);
  });
  it('b shifts the output without changing the step triangle', () => {
    expect(guess(3, 2, 2)).toBe(8);
    expect(stepTriangle(3, 0, 2).rise).toBeCloseTo(3);
    expect(stepTriangle(3, 2, 2).rise).toBeCloseTo(3);
    expect(stepTriangle(5, 0, 2).rise).toBeCloseTo(5);
  });
  it('two inputs: 3·2 + 2·1 + 0 = 8', () => {
    expect(neuronOut([W0, W_PEN], B0, [NOTEBOOKS, PENS])).toBe(8);
    expect(shares([W0, W_PEN], [NOTEBOOKS, PENS])).toEqual([6, 2]);
  });
});
