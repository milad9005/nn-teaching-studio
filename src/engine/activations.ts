// Activation functions and their derivatives.
// Convention: ReLU'(0) = 0 (same as PyTorch). ReLU is not differentiable at 0;
// any value in [0, 1] is a valid subgradient, and the engine always picks 0.

export type ActivationName = 'linear' | 'relu' | 'sigmoid' | 'tanh' | 'softmax';

export const ELEMENTWISE: ActivationName[] = ['linear', 'relu', 'sigmoid', 'tanh'];

export const sigmoid = (z: number) =>
  z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));

export function act(name: ActivationName, z: number): number {
  switch (name) {
    case 'linear':
      return z;
    case 'relu':
      return z > 0 ? z : 0;
    case 'sigmoid':
      return sigmoid(z);
    case 'tanh':
      return Math.tanh(z);
    case 'softmax':
      throw new Error('softmax is not elementwise; use activateLayer');
  }
}

/** Derivative of an elementwise activation, written in terms of z. */
export function actDeriv(name: ActivationName, z: number): number {
  switch (name) {
    case 'linear':
      return 1;
    case 'relu':
      return z > 0 ? 1 : 0;
    case 'sigmoid': {
      const s = sigmoid(z);
      return s * (1 - s);
    }
    case 'tanh': {
      const t = Math.tanh(z);
      return 1 - t * t;
    }
    case 'softmax':
      throw new Error('softmax derivative is a Jacobian; use backLayer');
  }
}

export function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((p, v) => p + v, 0);
  return e.map((v) => v / s);
}

export function activateLayer(name: ActivationName, z: number[]): number[] {
  return name === 'softmax' ? softmax(z) : z.map((v) => act(name, v));
}

/** Given dL/da for a layer, return dL/dz. Handles the softmax Jacobian. */
export function backLayer(name: ActivationName, z: number[], a: number[], da: number[]): number[] {
  if (name === 'softmax') {
    const dot = a.reduce((p, ai, i) => p + ai * da[i], 0);
    return a.map((ai, i) => ai * (da[i] - dot));
  }
  return z.map((zi, i) => da[i] * actDeriv(name, zi));
}

/** True when z sits on (or within tol of) a point where the activation is not differentiable. */
export function isKink(name: ActivationName, z: number, tol = 0): boolean {
  return name === 'relu' && Math.abs(z) <= tol;
}
