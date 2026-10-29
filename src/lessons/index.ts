// Lesson registry, in teaching order.
import { lesson01 } from './l01-inputs-outputs';
import { lesson02 } from './l02-weight-bias';
import { lesson03 } from './l03-activations';
import { lesson04 } from './l04-multiple-neurons';
import { lesson05 } from './l05-forward';
import { lesson06 } from './l06-loss';
import { lesson07 } from './l07-derivatives';
import { lesson08 } from './l08-loss-wrt-weight';
import { lesson09 } from './l09-partials';
import { lesson10 } from './l10-chain-rule';
import { lesson11 } from './l11-backprop';
import { lesson12 } from './l12-gradient-descent';
import { lesson13 } from './l13-learning-rate';
import { lesson14 } from './l14-batches';
import { lesson15 } from './l15-optimizers';
import { lesson16 } from './l16-initialization';
import { lesson17 } from './l17-overfitting';
import { lesson18 } from './l18-train-vs-inference';
import type { LessonDef } from './types';

export const LESSONS: LessonDef[] = [
  lesson01,
  lesson02,
  lesson03,
  lesson04,
  lesson05,
  lesson06,
  lesson07,
  lesson08,
  lesson09,
  lesson10,
  lesson11,
  lesson12,
  lesson13,
  lesson14,
  lesson15,
  lesson16,
  lesson17,
  lesson18,
];
