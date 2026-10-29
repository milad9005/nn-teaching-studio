import { describe, expect, it } from 'vitest';
import { COMPARE, errorOf, guess, lossOf, meanLoss, mirrorGuess, pourShape, receiptRows, W0, X, Y } from './math-loss';

describe('loss lesson math', () => {
  it('the running example: guess 6, error −4, loss 16', () => {
    const yhat = guess(W0, X);
    expect(yhat).toBe(6);
    expect(errorOf(yhat, Y)).toBe(-4);
    expect(lossOf(yhat, Y)).toBe(16);
  });
  it('4 too few or 4 too many cost the same', () => {
    expect(mirrorGuess(6, 10)).toBe(14);
    expect(lossOf(14, 10)).toBe(16);
    expect(COMPARE.map((e) => lossOf(10 - e, 10))).toEqual([1, 16, 64]);
  });
  it('4 receipts at w = 3: losses 9, 16, 36, 49, average 27.5', () => {
    expect(receiptRows(3).map((r) => r.loss)).toEqual([9, 16, 36, 49]);
    expect(meanLoss(3)).toBe(27.5);
    expect(receiptRows(3)[1]).toMatchObject({ x: 2, y: 10, yhat: 6, err: -4 });
  });
  it('the receipts are about 5 per notebook', () => {
    expect(receiptRows(5).map((r) => Math.abs(r.err))).toEqual([1, 0, 0, 1]);
    expect(meanLoss(5)).toBeLessThan(meanLoss(4.5));
    expect(meanLoss(5)).toBeLessThan(meanLoss(5.5));
  });
  it('pouring keeps the area, then levels every column to the average', () => {
    const sq = pourShape(36, 7, 27.5, 0, 0);
    expect(sq.width).toBeCloseTo(6);
    expect(sq.height).toBeCloseTo(6);
    const col = pourShape(36, 7, 27.5, 1, 0);
    expect(col.width * col.height).toBeCloseTo(36);
    const flat = pourShape(36, 7, 27.5, 1, 1);
    expect(flat.width * flat.height).toBeCloseTo(27.5);
  });
});
