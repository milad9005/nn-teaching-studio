// Lesson 11 math: phase of the forward/backward stepper and the cost of computing gradients
// (finite differences vs. backpropagation), counted in multiply-adds from the layer sizes.

import type { Stepper } from '../../state/lab';

export type Phase = 'idle' | 'forward' | 'loss' | 'backward' | 'update';

/** Which of the four training phases the lab stepper is showing. L = number of weight layers. */
export function phaseOf(stepper: Stepper, L: number): Phase {
  if (!stepper) return 'idle';
  if (stepper.phase === 'update') return 'update';
  if (stepper.phase === 'backward') return 'backward';
  return stepper.k >= L + 2 ? 'loss' : 'forward';
}

/** Number of trainable parameters (weights + biases) of a dense net with these layer sizes. */
export const paramCount = (sizes: number[]) => sizes.slice(1).reduce((s, n, l) => s + n * sizes[l] + n, 0);

/** Multiply-adds of one forward pass (one sample). */
export const forwardMacs = (sizes: number[]) => sizes.slice(1).reduce((s, n, l) => s + n * sizes[l], 0);

/**
 * Multiply-adds of one backward pass: every layer forms dW = dz·aᵀ (n_l·n_{l+1}), and every layer
 * except the first passes the gradient back, da = Wᵀ·dz (n_l·n_{l+1}). dL/dx is not needed.
 */
export const backwardMacs = (sizes: number[]) => {
  const F = forwardMacs(sizes);
  return F + (F - sizes[0] * (sizes[1] ?? 0));
};

export function gradientCost(sizes: number[]) {
  const P = paramCount(sizes);
  const F = forwardMacs(sizes);
  const B = backwardMacs(sizes);
  return {
    P,
    forward: F,
    /** One-sided differences: one base pass plus one pass per parameter. */
    fdForward: (P + 1) * F,
    /** Central differences (gradient check): two passes per parameter. */
    fdCentral: 2 * P * F,
    /** One forward pass and one backward pass, whatever P is. */
    backprop: F + B,
  };
}

/** Number of distinct paths from one unit of activation layer `layer` (1..L) to the output. */
export const pathCount = (sizes: number[], layer: number) => sizes.slice(layer + 1).reduce((p, n) => p * n, 1);
