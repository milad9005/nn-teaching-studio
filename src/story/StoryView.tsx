// Lesson mode: one big visual, one short sentence under it, Next / Previous.
// The journey bar on top keeps the whole road in view; the side panel holds the fuller
// written explanation for people learning on their own.

import { useEffect, useLayoutEffect, useState } from 'react';
import { useT } from '../i18n';
import { useUi } from '../state/ui';
import { useLab } from '../state/lab';
import { resetLesson } from '../state/lessonState';
import { withMath } from '../components/prose';
import { CURRICULUM, lessonIndexById } from './curriculum';
import { LESSONS } from './index';

const typing = (e: KeyboardEvent) => {
  const el = e.target as HTMLElement;
  return el.tagName === 'INPUT' || el.tagName === 'SELECT' || el.tagName === 'TEXTAREA' || el.isContentEditable;
};

/** Next / previous step, crossing into the next or previous lesson at the ends. */
export function stepBy(d: 1 | -1) {
  const ui = useUi.getState();
  const lesson = LESSONS[ui.lessonIndex];
  const n = lesson.steps.length;
  if (d === 1) {
    if (ui.step < n - 1) ui.set({ step: ui.step + 1 });
    else if (ui.lessonIndex < LESSONS.length - 1) {
      ui.set({ completed: { ...ui.completed, [lesson.id]: true } });
      ui.goLesson(ui.lessonIndex + 1);
    }
  } else if (ui.step > 0) ui.set({ step: ui.step - 1 });
  else if (ui.lessonIndex > 0) {
    const prev = LESSONS[ui.lessonIndex - 1];
    ui.set({ lessonIndex: ui.lessonIndex - 1, step: prev.steps.length - 1 });
  }
}

/** Arrow keys and space move through steps (in all modes except the playground). */
function useStepKeys() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const ui = useUi.getState();
      if (ui.mode !== 'guided' || typing(e) || e.metaKey || e.ctrlKey || e.altKey) return;
      const fa = ui.lang === 'fa';
      if (e.key === ' ' || e.key === (fa ? 'ArrowLeft' : 'ArrowRight')) {
        e.preventDefault();
        stepBy(1);
      } else if (e.key === (fa ? 'ArrowRight' : 'ArrowLeft')) {
        e.preventDefault();
        stepBy(-1);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}

export function JourneyBar() {
  const { tr } = useT();
  const idx = useUi((s) => s.lessonIndex);
  const completed = useUi((s) => s.completed);
  const goLesson = useUi((s) => s.goLesson);
  return (
    <nav className="journey" aria-label="course" data-testid="journey">
      {CURRICULUM.map((l, i) => (
        <button
          key={l.id}
          className={`journey-stop ${i === idx ? 'on' : ''} ${i < idx || completed[l.id] ? 'done' : ''}`}
          onClick={() => goLesson(i)}
          aria-current={i === idx ? 'step' : undefined}
          title={tr(l.title)}
        >
          <span className="journey-dot">{i + 1}</span>
          <span className="journey-name">{tr(l.short)}</span>
        </button>
      ))}
    </nav>
  );
}

export function StoryView() {
  const { t, tr } = useT();
  const idx = useUi((s) => Math.min(s.lessonIndex, LESSONS.length - 1));
  const rawStep = useUi((s) => s.step);
  const explainOpen = useUi((s) => s.explainOpen);
  const captions = useUi((s) => s.captions);
  const presentation = useUi((s) => s.presentation);
  const toggle = useUi((s) => s.toggle);
  const set = useUi((s) => s.set);
  const goLesson = useUi((s) => s.goLesson);
  const lesson = LESSONS[idx];
  const step = Math.max(0, Math.min(rawStep, lesson.steps.length - 1));
  const cur = lesson.steps[step];
  useStepKeys();

  // Entering a lesson loads its preset; the scene renders only after that.
  const [loadedFor, setLoadedFor] = useState<string | null>(null);
  useLayoutEffect(() => {
    const lab = useLab.getState();
    lab.setRunning(false);
    lab.showAll();
    lab.select(null);
    if (lesson.presetId) lab.loadPreset(lesson.presetId);
    setLoadedFor(lesson.id);
  }, [lesson.id, lesson.presetId]);
  const ready = loadedFor === lesson.id;

  const restart = () => {
    resetLesson(lesson.id);
    useLab.getState().setRunning(false);
    if (lesson.presetId) useLab.getState().loadPreset(lesson.presetId);
    set({ step: 0 });
  };
  const last = step === lesson.steps.length - 1;
  const recallIdx = cur.recall ? lessonIndexById.get(cur.recall.lesson) : undefined;
  const { Scene } = lesson;

  return (
    <div className={`story ${explainOpen && !presentation ? 'with-explain' : ''}`} data-testid="story">
      {!presentation && <JourneyBar />}
      <div className="story-body">
        <section className="story-stage" aria-label={tr(lesson.title)}>
          <header className="story-head">
            <span className="story-num num">{String(idx + 1).padStart(2, '0')}</span>
            <h1 className="story-title" data-testid="lesson-title">
              {tr(lesson.title)}
            </h1>
          </header>
          <div className="scene" data-testid="scene" data-step={step}>
            {ready && <Scene step={step} />}
          </div>
          <div className="caption-box" data-testid="caption-box">
            <div className="step-dots" aria-hidden="true">
              {lesson.steps.map((_, i) => (
                <button key={i} className={`step-dot ${i === step ? 'on' : i < step ? 'done' : ''}`} onClick={() => set({ step: i })} tabIndex={-1} />
              ))}
            </div>
            {(captions || !presentation) && (
              <p className="caption" data-testid="caption" key={`${lesson.id}-${step}`}>
                {withMath(tr(cur.say))}
              </p>
            )}
            {cur.recall && recallIdx !== undefined && (captions || !presentation) && (
              <button className="recall-chip" onClick={() => goLesson(recallIdx)} data-testid="recall">
                ↺ {t('recall.from')} {recallIdx + 1} · {tr(CURRICULUM[recallIdx].short)}: <span>{withMath(tr(cur.recall.note))}</span>
              </button>
            )}
            <nav className="step-nav">
              <button className="btn" onClick={() => stepBy(-1)} disabled={idx === 0 && step === 0} data-testid="step-prev">
                {t('nav.prev')}
              </button>
              <span className="step-count num" data-testid="step-count">
                {step + 1} / {lesson.steps.length}
              </span>
              <button className="btn primary" onClick={() => stepBy(1)} disabled={idx === LESSONS.length - 1 && last} data-testid="step-next">
                {last && idx < LESSONS.length - 1 ? `${t('story.nextLesson')} · ${tr(CURRICULUM[idx + 1].short)}` : t('nav.next')}
              </button>
              <span className="grow" />
              <button className="btn ghost sm" onClick={restart} title={t('nav.reset')} data-testid="lesson-restart">
                ↺
              </button>
              {!presentation && (
                <button className={`btn ghost sm ${explainOpen ? 'active' : ''}`} aria-pressed={explainOpen} onClick={() => toggle('explainOpen')} data-testid="explain-toggle">
                  {t('story.explain')}
                </button>
              )}
            </nav>
          </div>
        </section>
        {explainOpen && !presentation && (
          <aside className="story-explain" data-testid="explain-panel">
            <h2>{t('story.explain')}</h2>
            <ol className="transcript">
              {lesson.steps.slice(0, step + 1).map((s, i) => (
                <li key={i} className={i === step ? 'on' : ''} onClick={() => set({ step: i })}>
                  <p className="t-say">{withMath(tr(s.say))}</p>
                  {i === step && s.more && <p className="t-more">{withMath(tr(s.more))}</p>}
                </li>
              ))}
            </ol>
            {!last && <p className="muted small">{t('story.more')}</p>}
          </aside>
        )}
      </div>
    </div>
  );
}
