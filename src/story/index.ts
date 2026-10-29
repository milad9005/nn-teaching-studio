// Lesson registry: every file in ./lessons exports one StoryLesson; order comes from curriculum.ts.
import { createElement } from 'react';
import { CURRICULUM, type LessonMeta } from './curriculum';
import type { StoryLesson } from './types';
import { bi } from '../i18n';

const modules = import.meta.glob<Record<string, unknown>>('./lessons/*.tsx', { eager: true });

const isLesson = (v: unknown): v is StoryLesson => typeof v === 'object' && v !== null && 'id' in v && 'steps' in v && 'Scene' in v;

export const BODIES: Map<string, StoryLesson> = new Map(
  Object.values(modules)
    .flatMap((m) => Object.values(m))
    .filter(isLesson)
    .map((l) => [l.id, l]),
);

/** Placeholder while a lesson file does not exist yet; story.test.ts fails in that case. */
const missing = (id: string): StoryLesson => ({
  id,
  steps: [{ say: bi(`Lesson "${id}" is not written yet.`, `درس «${id}» هنوز نوشته نشده.`) }],
  Scene: () => createElement('div', { className: 'muted' }, '…'),
});

export type LessonDef = LessonMeta & StoryLesson;
export const LESSONS: LessonDef[] = CURRICULUM.map((meta) => ({ ...(BODIES.get(meta.id) ?? missing(meta.id)), ...meta }));
