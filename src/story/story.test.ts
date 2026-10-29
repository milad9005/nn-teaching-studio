// The course as a whole: every lesson exists, every step is written in both languages,
// callbacks only point backwards, and no notation beyond what the story teaches appears.
import { CURRICULUM, lessonIndexById } from './curriculum';
import { BODIES, LESSONS } from './index';

const FORBIDDEN: [RegExp, string][] = [
  [/\\theta|θ/, 'θ'],
  [/\\epsilon|\\varepsilon|ε/, 'ε'],
  [/\\sqrt|√/, '√'],
  [/\\log\b|\\ln\b/, 'log'],
  [/\\lim\b/, 'lim'],
  [/\\operatorname\{Var\}|variance/i, 'variance'],
  [/\d\s?%/, 'percent'],
  [/bmatrix|\\top\b/, 'matrix'],
];

describe('course structure', () => {
  it('has 12 lessons, each with a lesson file', () => {
    expect(CURRICULUM).toHaveLength(12);
    for (const l of CURRICULUM) expect(BODIES.has(l.id), `missing lesson file for ${l.id}`).toBe(true);
    expect(LESSONS.map((l) => l.id)).toEqual(CURRICULUM.map((l) => l.id));
  });
});

describe.each(CURRICULUM.map((l, i) => [l.id, i] as const))('lesson %s', (id, i) => {
  const lesson = BODIES.get(id);

  it('has a short sequence of steps', () => {
    expect(lesson).toBeDefined();
    expect(lesson!.steps.length).toBeGreaterThanOrEqual(5);
    expect(lesson!.steps.length).toBeLessThanOrEqual(12);
  });

  it('every step is written in Persian and English, with a fuller explanation', () => {
    lesson!.steps.forEach((s, k) => {
      expect(s.say.fa.trim(), `step ${k + 1} fa`).not.toBe('');
      expect(s.say.en.trim(), `step ${k + 1} en`).not.toBe('');
      expect(s.more, `step ${k + 1} more`).toBeDefined();
      expect(s.more!.fa.trim()).not.toBe('');
      expect(s.more!.en.trim()).not.toBe('');
      // Captions stay short enough to read while the visual changes.
      expect(s.say.en.length, `step ${k + 1} caption too long`).toBeLessThan(260);
    });
  });

  it('only recalls earlier lessons', () => {
    for (const s of lesson!.steps) {
      if (!s.recall) continue;
      const j = lessonIndexById.get(s.recall.lesson);
      expect(j, `unknown lesson ${s.recall.lesson}`).toBeDefined();
      expect(j!).toBeLessThan(i);
    }
  });

  it('uses no notation the course does not teach', () => {
    const text = lesson!.steps.flatMap((s) => [s.say.en, s.say.fa, s.more?.en ?? '', s.more?.fa ?? '', s.recall?.note.en ?? '', s.recall?.note.fa ?? '']).join('\n');
    for (const [re, name] of FORBIDDEN) expect(re.test(text), `uses ${name}`).toBe(false);
  });
});

describe('the story connects', () => {
  it('later lessons call back to earlier ones', () => {
    const recalls = [...BODIES.values()].flatMap((l) => l.steps.filter((s) => s.recall).map((s) => s.recall!.lesson));
    // The core ideas are recalled where they come back: the slope (lesson 2) and the derivative (lesson 5).
    expect(recalls).toContain('neuron');
    expect(recalls).toContain('derivative');
    expect(recalls.length).toBeGreaterThanOrEqual(6);
  });
});
