// Weight initialization schemes. All are seeded; biases start at 0.
//   zeros:   every weight 0 (hidden units stay identical forever: symmetry is never broken)
//   uniform: U(-scale, scale)
//   xavier:  U(-r, r), r = sqrt(6 / (fanIn + fanOut))      (Glorot & Bengio, 2010)
//   he:      N(0, sqrt(2 / fanIn))                          (He et al., 2015; suited to ReLU)

import { mulberry32, normal, uniform } from './rng';
import { cloneNet, type Network } from './network';

export type InitName = 'zeros' | 'constant' | 'uniform' | 'xavier' | 'he';

export function initialize(net: Network, name: InitName, seed = 1, scale = 1): Network {
  const rng = mulberry32(seed);
  const n = cloneNet(net);
  n.layers.forEach((layer) => {
    const fanIn = layer.W[0]?.length ?? 1;
    const fanOut = layer.W.length;
    layer.W = layer.W.map((row) =>
      row.map(() => {
        switch (name) {
          case 'zeros':
            return 0;
          case 'constant':
            return scale;
          case 'uniform':
            return uniform(rng, -scale, scale);
          case 'xavier': {
            const r = Math.sqrt(6 / (fanIn + fanOut));
            return uniform(rng, -r, r);
          }
          case 'he':
            return normal(rng, 0, Math.sqrt(2 / fanIn));
        }
      }),
    );
    layer.b = layer.b.map(() => 0);
  });
  return n;
}
