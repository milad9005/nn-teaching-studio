import { describe, expect, it } from 'vitest';
import {
  actDeriv,
  batchLossGrad,
  backward,
  forward,
  gradCheck,
  lossGrad,
  lossSample,
  makeNetwork,
  maxRelErr,
  pathsFrom,
  sumContrib,
  sampleResult,
  softmax,
  backLayer,
  addUnit,
  removeUnit,
  addLayer,
  removeLayer,
  sizes,
  validate,
  paramRefs,
  initialize,
  centralDiff,
  type LossName,
  type Network,
  type ActivationName,
} from './index';
import { mulberry32, uniform } from './rng';

const close = (a: number, b: number, tol = 1e-9) => expect(Math.abs(a - b)).toBeLessThan(tol);

describe('core teaching example: two sequential ReLU neurons', () => {
  // x = 2, w1 = 3, w2 = 2, b = 0, y = 10
  const net = makeNetwork([1, 1, 1], ['relu', 'relu'], (l) => (l === 0 ? 3 : 2));
  const r = sampleResult(net, [2], [10], 'mse');

  it('forward values', () => {
    expect(r.trace.z[0][0]).toBe(6);
    expect(r.trace.a[1][0]).toBe(6);
    expect(r.trace.z[1][0]).toBe(12);
    expect(r.trace.a[2][0]).toBe(12);
    expect(r.loss).toBe(4);
  });

  it('backward values', () => {
    expect(r.dOut[0]).toBe(4); // dL/doutput = 2(12-10)
    expect(r.back.grads.W[0][0][0]).toBe(16); // 4 * 2 * 2
    expect(r.back.grads.W[1][0][0]).toBe(24); // 4 * 6
    expect(r.back.grads.b[0][0]).toBe(8); // 4 * 2 * 1
    expect(r.back.grads.b[1][0]).toBe(4);
    expect(r.back.da[1][0]).toBe(8); // dL/da1 = 4 * w2
  });

  it('chain-rule path for w1 multiplies to 16', () => {
    const paths = pathsFrom(net, r.trace, r.back.dz, 0, 0, r.trace.a[0][0]);
    expect(paths).toHaveLength(1);
    expect(paths[0].contribution).toBe(16);
  });

  it('dead first neuron: w1 < 0 makes every gradient upstream 0', () => {
    const dead = makeNetwork([1, 1, 1], ['relu', 'relu'], (l) => (l === 0 ? -1 : 2));
    const d = sampleResult(dead, [2], [10], 'mse');
    expect(d.trace.a[2][0]).toBe(0);
    expect(d.loss).toBe(100);
    expect(d.back.grads.W[0][0][0]).toBe(0);
    expect(Math.abs(d.back.grads.W[1][0][0])).toBe(0); // a1 = 0
  });

  it('ReLU derivative convention at exactly z = 0 is 0', () => {
    expect(actDeriv('relu', 0)).toBe(0);
    const edge = makeNetwork([1, 1, 1], ['relu', 'relu'], (l) => (l === 0 ? 0 : 2));
    const e = sampleResult(edge, [2], [10], 'mse');
    expect(e.trace.z[0][0]).toBe(0);
    expect(e.back.grads.W[0][0][0]).toBe(0);
  });
});

describe('branching graph: contributions add where paths meet', () => {
  const net: Network = {
    inputSize: 1,
    layers: [
      { W: [[1], [0.5]], b: [0, 0], act: 'relu' },
      { W: [[1, 1], [0.5, -1]], b: [0, 0.5], act: 'relu' },
      { W: [[1, 2]], b: [0], act: 'linear' },
    ],
  };
  const r = sampleResult(net, [1], [1], 'mse');
  it('forward and gradients', () => {
    expect(r.trace.a[3][0]).toBe(2.5);
    expect(r.loss).toBe(2.25);
    expect(r.back.dz[1]).toEqual([3, 6]);
    expect(r.back.grads.W[0]).toEqual([[6], [-3]]);
  });
  it('two paths from W1[0,0] each contribute 3', () => {
    const paths = pathsFrom(net, r.trace, r.back.dz, 0, 0, 1);
    expect(paths.map((p) => p.contribution)).toEqual([3, 3]);
    expect(sumContrib(paths)).toBe(r.back.grads.W[0][0][0]);
  });
});

describe('path sums equal backprop for random nets', () => {
  for (const act of ['tanh', 'sigmoid', 'relu'] as ActivationName[]) {
    it(act, () => {
      const rng = mulberry32(5);
      const net = makeNetwork([2, 3, 3, 2], [act, act, 'linear'], () => uniform(rng, -1, 1), () => uniform(rng, -0.5, 0.5));
      const r = sampleResult(net, [0.7, -0.4], [0.2, -0.1], 'mse');
      for (let l = 0; l < 3; l++)
        for (let j = 0; j < net.layers[l].b.length; j++) close(sumContrib(pathsFrom(net, r.trace, r.back.dz, l, j)), r.back.dz[l][j]);
    });
  }
});

describe('finite-difference gradient checks', () => {
  const cases: { loss: LossName; out: ActivationName; Y: number[][]; outN: number }[] = [
    { loss: 'mse', out: 'linear', Y: [[0.3], [-0.5], [1.2]], outN: 1 },
    { loss: 'mae', out: 'linear', Y: [[0.3], [-0.5], [1.2]], outN: 1 },
    { loss: 'huber', out: 'linear', Y: [[3], [-0.5], [1.2]], outN: 1 },
    { loss: 'bce', out: 'sigmoid', Y: [[1], [0], [1]], outN: 1 },
    { loss: 'cce', out: 'softmax', Y: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], outN: 3 },
  ];
  for (const c of cases) {
    for (const hidden of ['tanh', 'sigmoid', 'relu'] as ActivationName[]) {
      it(`${c.loss} / ${hidden}`, () => {
        const net = initialize(makeNetwork([2, 4, 3, c.outN], [hidden, hidden, c.out]), 'xavier', 11);
        const data = { X: [[0.5, -1], [1.5, 0.3], [-0.7, 0.9]], Y: c.Y };
        const rows = gradCheck(net, data, c.loss, { delta: 1 });
        expect(maxRelErr(rows)).toBeLessThan(1e-5);
      });
    }
  }

  it('mini-batch gradient is the mean of per-sample gradients', () => {
    const net = initialize(makeNetwork([1, 3, 1], ['tanh', 'linear']), 'xavier', 2);
    const data = { X: [[0.1], [0.5], [-1]], Y: [[1], [0], [2]] };
    const full = batchLossGrad(net, data, 'mse').grads;
    const parts = [0, 1, 2].map((i) => batchLossGrad(net, data, 'mse', {}, [i]).grads);
    close(full.W[0][1][0], (parts[0].W[0][1][0] + parts[1].W[0][1][0] + parts[2].W[0][1][0]) / 3);
  });
});

describe('losses', () => {
  it('values', () => {
    expect(lossSample('mse', [12], [10])).toBe(4);
    expect(lossSample('mae', [12], [10])).toBe(2);
    expect(lossSample('huber', [12], [10], { delta: 1 })).toBe(1.5);
    expect(lossSample('huber', [10.5], [10], { delta: 1 })).toBe(0.125);
    close(lossSample('bce', [0.8], [1]), -Math.log(0.8));
    close(lossSample('cce', [0.7, 0.2, 0.1], [0, 1, 0]), -Math.log(0.2));
  });
  it('gradients match central differences', () => {
    for (const name of ['mse', 'mae', 'huber', 'bce'] as LossName[]) {
      const y = name === 'bce' ? [1] : [0.4];
      for (const p of [0.13, 0.62, 0.9]) close(lossGrad(name, [p], y, { delta: 0.3 })[0], centralDiff((v) => lossSample(name, [v], y, { delta: 0.3 }), p), 1e-5);
    }
  });
  it('softmax Jacobian', () => {
    const z = [0.2, -1, 1.3];
    const a = softmax(z);
    const da = [0.5, -2, 1];
    const dz = backLayer('softmax', z, a, da);
    dz.forEach((d, i) => {
      const f = (t: number) => {
        const zz = z.slice();
        zz[i] = t;
        return softmax(zz).reduce((s, v, k) => s + v * da[k], 0);
      };
      close(d, centralDiff(f, z[i]), 1e-7);
    });
  });
});

describe('architecture edits', () => {
  const net = makeNetwork([2, 3, 1], ['relu', 'linear'], () => 0.3);
  it('add/remove units keeps shapes valid', () => {
    const a = addUnit(net, 0);
    expect(sizes(a)).toEqual([2, 4, 1]);
    expect(validate(a)).toEqual([]);
    const r = removeUnit(a, 0, 1);
    expect(sizes(r)).toEqual([2, 3, 1]);
    expect(validate(r)).toEqual([]);
  });
  it('add/remove layers keeps shapes valid and parameters in sync', () => {
    const a = addLayer(net, 1, 2, 'tanh');
    expect(sizes(a)).toEqual([2, 3, 2, 1]);
    expect(validate(a)).toEqual([]);
    expect(paramRefs(a)).toHaveLength(2 * 3 + 3 + 3 * 2 + 2 + 2 + 1);
    const r = removeLayer(a, 1);
    expect(sizes(r)).toEqual([2, 3, 1]);
    expect(validate(r)).toEqual([]);
    expect(() => forward(r, [1, 2])).not.toThrow();
  });
  it('validation catches bad dimensions', () => {
    const errs = validate(net, { X: [[1, 2, 3]], Y: [[1]] });
    expect(errs[0]).toMatchObject({ code: 'inputDim', expected: 2, got: 3 });
  });
  it('backward with a non-unit scale divides gradients', () => {
    const t = forward(net, [1, 1]);
    const g1 = backward(net, t, [2]).grads.W[1][0][0];
    const g2 = backward(net, t, [2], 0.5).grads.W[1][0][0];
    close(g2, g1 / 2);
  });
});

describe('number formatting', () => {
  it('rounds float noise to 0 and keeps real small values', async () => {
    const { fmt, tex } = await import('./numeric');
    expect(fmt(-8.86e-15)).toBe('0');
    expect(tex(-8.86e-15)).toBe('0');
    expect(fmt(-0.5)).toBe('−0.5');
    expect(fmt(2e-6, 3)).toContain('e');
  });
});

describe('trailing zeros', () => {
  it('keeps integer zeros', async () => {
    const { fmt, tex } = await import('./numeric');
    expect(fmt(100, 0)).toBe('100');
    expect(tex(250, 0)).toBe('250');
    expect(fmt(1.5, 3)).toBe('1.5');
    expect(fmt(2, 3)).toBe('2');
  });
});
