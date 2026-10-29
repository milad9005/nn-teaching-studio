import type { ComponentType } from 'react';
import type { Bi } from '../i18n';
import type { PresetId } from '../engine';

/** One step of a lesson: a short sentence under the visual, and what the visual shows. */
export interface StoryStep {
  /** The narration under the visual: one or two short sentences, Persian written first. */
  say: Bi;
  /** A fuller explanation for people reading on their own (the side panel). Optional. */
  more?: Bi;
  /** Points back to an earlier lesson that this step builds on. */
  recall?: { lesson: string; note: Bi };
}

/** What a lesson file in src/story/lessons exports. Title and order live in curriculum.ts. */
export interface StoryLesson {
  /** Matches an id in CURRICULUM. */
  id: string;
  steps: StoryStep[];
  /** The visual. Draws the state for `step` (0-based) and animates between steps. */
  Scene: ComponentType<{ step: number }>;
  /** Loaded into the shared lab store when the lesson opens (network lessons). */
  presetId?: PresetId;
}
