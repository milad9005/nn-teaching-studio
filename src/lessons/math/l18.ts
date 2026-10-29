// Lesson 18 math: what one training iteration and one inference call cost, counted from the
// layer sizes of a dense network (multiply-adds; activations and the loss are a few extra ops).

export interface Cost {
  params: number;
  /** Multiply-adds of one forward pass for one example: sum of n_in · n_out. */
  forward: number;
  /** One backward pass for one example: dL/dW (n_in · n_out) plus dL/da for the layer below. */
  backward: number;
  /** One optimizer update touches every parameter once. */
  update: number;
}

export function costOf(sizes: number[]): Cost {
  let params = 0;
  let forward = 0;
  let backward = 0;
  for (let l = 0; l + 1 < sizes.length; l++) {
    const mac = sizes[l] * sizes[l + 1];
    params += mac + sizes[l + 1];
    forward += mac;
    // Weight gradients always; input gradients only below the first layer are needed.
    backward += mac + (l > 0 ? mac : 0);
  }
  return { params, forward, backward, update: params };
}

/** Operations for one training iteration on a batch of B examples vs. one inference call. */
export function iterationCost(sizes: number[], B: number) {
  const c = costOf(sizes);
  return { training: B * (c.forward + c.backward) + c.update, inference: c.forward, cost: c };
}
