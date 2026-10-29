import { describe, expect, it } from 'vitest';
import { distinctUnits, ellipseAt, momentumParts, output, pathOf, RACE_STEPS, runOf, showcaseStep, stepsToReach, valleyLoss, zigzags, BETA } from './math-better';

describe('lesson 11 · better', () => {
  it('contour ellipses sit exactly on their loss level', () => {
    const [a, b] = ellipseAt(2);
    expect(valleyLoss(a, 0)).toBeCloseTo(2, 10);
    expect(valleyLoss(0, b)).toBeCloseTo(2, 10);
  });

  it('plain steps zig-zag; momentum turns back far less and gets lower', () => {
    const sgd = pathOf('sgd');
    const mom = pathOf('momentum');
    expect(sgd.points.length).toBe(RACE_STEPS + 1);
    expect(zigzags(sgd)).toBeGreaterThan(40);
    expect(zigzags(mom)).toBeLessThan(zigzags(sgd) / 3);
    expect(mom.losses[RACE_STEPS]).toBeLessThan(sgd.losses[RACE_STEPS] / 5);
  });

  it('Adam reaches the bottom area in fewer steps than plain steps', () => {
    const adam = stepsToReach(pathOf('adam'), 0.05);
    expect(adam).toBeGreaterThan(0);
    expect(stepsToReach(pathOf('sgd'), 0.05)).toBe(-1);
  });

  it('each momentum step = 0.9 × the previous step + the new slope push (as the engine moves)', () => {
    for (const k of [1, 5, showcaseStep()]) {
      const m = momentumParts(k);
      expect(m.kept[0]).toBeCloseTo(BETA * m.prev[0], 12);
      expect(m.kept[0] + m.push[0]).toBeCloseTo(m.next[0], 10);
      expect(m.kept[1] + m.push[1]).toBeCloseTo(m.next[1], 10);
    }
  });

  it('the showcase step keeps going along the valley while the slope pushes across it', () => {
    const m = momentumParts(showcaseStep());
    expect(Math.abs(m.kept[0])).toBeGreaterThan(Math.abs(m.kept[1]));
    expect(Math.abs(m.push[1])).toBeGreaterThan(Math.abs(m.push[0]));
  });

  it('identical starts stay copies; random starts separate and fit the wave', () => {
    const same = runOf('same');
    const rnd = runOf('random');
    expect(same.units.every((u) => distinctUnits(u) === 1)).toBe(true);
    expect(distinctUnits(rnd.units[rnd.units.length - 1])).toBe(4);
    const end = (r: typeof same) => r.loss[r.loss.length - 1];
    expect(end(rnd)).toBeLessThan(end(same) / 10);
    const k = rnd.units.length - 1;
    expect(Number.isFinite(output(rnd.units[k], rnd.outB[k], 0.3))).toBe(true);
  });
});
