import type { ComponentType } from 'react';
import type { Bi } from '../i18n';
import type { PresetId } from '../engine';
import type { CheckpointQ } from '../components/teaching';

/**
 * A lesson fills the four workspace panels. All panels of a lesson must read the same state:
 * either the shared lab store (when `presetId` is set) or `useLessonState(id, ...)`.
 */
export interface LessonDef {
  /** "l01" ... "l18"; also the prefix for useLessonState and resetLesson. */
  id: string;
  title: Bi;
  /** Short label for the lesson list. */
  short: Bi;
  objectives: Bi[];
  /** Loaded into the shared lab store when the lesson opens and on Reset. */
  presetId?: PresetId;
  /** How many <Reveal at={n}> steps the lesson has (presentation mode). */
  revealSteps: number;
  /** Left panel: concept explanation, predictions, worked example. */
  Explain: ComponentType;
  /** Center panel: the main interactive visualization. */
  Center: ComponentType;
  /** Right panel: editable parameters, live equations, numbers. */
  Right?: ComponentType;
  /** Bottom panel: graphs, loss curves, history, timeline. */
  Bottom?: ComponentType;
  checkpoint?: CheckpointQ;
}
