import { useLayoutEffect, useState } from 'react';
import { LESSONS } from '../lessons';
import { useUi } from '../state/ui';
import { useLab } from '../state/lab';
import { resetLesson } from '../state/lessonState';
import { useT } from '../i18n';
import { Panel, Workspace } from './Workspace';
import { Checkpoint, withMath } from '../components/teaching';

export function useLessonReset() {
  const idx = useUi((s) => s.lessonIndex);
  const lesson = LESSONS[Math.min(idx, LESSONS.length - 1)];
  return () => {
    resetLesson(lesson.id);
    useLab.getState().setRunning(false);
    if (lesson.presetId) useLab.getState().loadPreset(lesson.presetId);
  };
}

export function GuidedMode() {
  const { t, tr } = useT();
  const idx = useUi((s) => s.lessonIndex);
  const goLesson = useUi((s) => s.goLesson);
  const lesson = LESSONS[Math.min(idx, LESSONS.length - 1)];
  const reset = useLessonReset();

  // Entering a lesson loads its preset into the shared lab (language switches do not re-run this).
  // The lesson's panels render only after that, so they never see the previous lesson's network.
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

  const { Explain, Center, Right, Bottom } = lesson;
  return (
    <Workspace
      left={
        <Panel id="left" title="panel.explain">
          <div className="lesson-head" data-testid="lesson-head">
            <div className="lesson-num">
              {t('nav.lesson')} {String(idx + 1).padStart(2, '0')} / {LESSONS.length}
            </div>
            <h1 className="lesson-title">{tr(lesson.title)}</h1>
            <div className="objectives">
              <div className="small muted">{t('nav.objectives')}</div>
              <ul>
                {lesson.objectives.map((o, i) => (
                  <li key={i}>{withMath(tr(o))}</li>
                ))}
              </ul>
            </div>
          </div>
          <div className="lesson-body" key={lesson.id}>
            {ready && <Explain />}
            {lesson.checkpoint && <Checkpoint id={lesson.id} q={lesson.checkpoint} />}
          </div>
          <nav className="lesson-nav">
            <button className="btn" disabled={idx === 0} onClick={() => goLesson(idx - 1)} data-testid="lesson-prev">
              {t('nav.prev')}
            </button>
            <button className="btn ghost" onClick={reset} data-testid="lesson-reset">
              ↺ {t('nav.reset')}
            </button>
            <button className="btn primary" disabled={idx === LESSONS.length - 1} onClick={() => goLesson(idx + 1)} data-testid="lesson-next">
              {t('nav.next')}
            </button>
          </nav>
        </Panel>
      }
      center={
        <Panel id="center" title="panel.visual" maximizable>
          {ready && <Center key={lesson.id} />}
        </Panel>
      }
      right={
        Right ? (
          <Panel id="right" title="panel.math" maximizable>
            {ready && <Right key={lesson.id} />}
          </Panel>
        ) : undefined
      }
      bottom={
        Bottom ? (
          <Panel id="bottom" title="panel.graphs">
            {ready && <Bottom key={lesson.id} />}
          </Panel>
        ) : undefined
      }
    />
  );
}
