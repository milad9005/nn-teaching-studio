// Building blocks for lesson prose: predictions, checkpoints, step reveal, advanced math.

import { useState, type ReactNode } from 'react';
import { useT, type Bi } from '../i18n';
import { useUi } from '../state/ui';
import { useLessonStore } from '../state/lessonState';
import { Tex } from './Tex';

/** Bilingual paragraph/inline text. */
export function T({ children }: { children: Bi }) {
  const { tr } = useT();
  return <>{tr(children)}</>;
}

/**
 * Text with inline math: write `{ en: 'If $w > 0$ ...', fa: '...' }` and $...$ becomes KaTeX.
 * Keeps prose and math in one string per language.
 */
export function P({ children, className }: { children: Bi | string; className?: string }) {
  const { tr } = useT();
  return <p className={className}>{withMath(tr(children))}</p>;
}
export function withMath(s: string): ReactNode[] {
  return s.split(/(\$[^$]+\$)/g).map((part, i) =>
    part.startsWith('$') && part.endsWith('$') && part.length > 2 ? <Tex key={i}>{part.slice(1, -1)}</Tex> : <span key={i}>{part}</span>,
  );
}

/**
 * Content revealed step by step in presentation mode. Outside presentation it is always shown.
 */
export function Reveal({ at, children }: { at: number; children: ReactNode }) {
  const presentation = useUi((s) => s.presentation);
  const reveal = useUi((s) => s.reveal);
  if (presentation && reveal < at) return <div className="reveal-pending" aria-hidden="true" />;
  return <div className={presentation ? 'reveal-in' : undefined}>{children}</div>;
}

/** Rigorous detail that the instructor can hide (key A in presentation mode). */
export function Advanced({ title, children, defaultOpen = false }: { title?: Bi; children: ReactNode; defaultOpen?: boolean }) {
  const { t, tr } = useT();
  const hide = useUi((s) => s.hideAdvanced);
  const [open, setOpen] = useState(defaultOpen);
  if (hide) return null;
  return (
    <div className={`advanced ${open ? 'open' : ''}`}>
      <button type="button" className="advanced-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span className="chev" aria-hidden="true">
          ▸
        </span>
        {title ? tr(title) : t('advanced.title')}
      </button>
      {open && <div className="advanced-body">{children}</div>}
    </div>
  );
}

export function Callout({ children, tone = 'info' }: { children: ReactNode; tone?: 'info' | 'warn' | 'err' }) {
  return <div className={`callout ${tone === 'info' ? '' : tone}`}>{children}</div>;
}

export interface Choice {
  id: string;
  label: Bi;
}

/**
 * Predict-then-verify prompt. The answer is stored per lesson so it survives language switches.
 * `explain` is shown after answering and should point at what to try in the demo.
 */
export function Predict({
  id,
  question,
  choices,
  answer,
  explain,
}: {
  id: string;
  question: Bi;
  choices: Choice[];
  answer: string;
  explain: Bi;
}) {
  const { t, tr } = useT();
  const key = `${id}:predict`;
  const picked = useLessonStore((s) => s.data[key]) as string | undefined;
  const setStore = useLessonStore((s) => s.set);
  return (
    <div className="predict" data-testid={`predict-${id}`}>
      <div className="predict-head">{t('predict.title')}</div>
      <div className="predict-q">{withMath(tr(question))}</div>
      <div className="predict-choices">
        {choices.map((c) => (
          <button
            key={c.id}
            type="button"
            className={`btn sm ${picked === c.id ? (c.id === answer ? 'right' : 'wrong') : ''}`}
            aria-pressed={picked === c.id}
            onClick={() => setStore(key, c.id)}
          >
            {withMath(tr(c.label))}
          </button>
        ))}
      </div>
      {picked && (
        <div className="predict-explain" role="status">
          <strong>{picked === answer ? t('predict.right') : t('predict.wrong')}</strong> {withMath(tr(explain))}
        </div>
      )}
    </div>
  );
}

export interface CheckpointQ {
  question: Bi;
  choices: Choice[];
  answer: string;
  explain: Bi;
}

export function Checkpoint({ id, q }: { id: string; q: CheckpointQ }) {
  const { t, tr } = useT();
  const key = `${id}:checkpoint`;
  const picked = useLessonStore((s) => s.data[key]) as string | undefined;
  const setStore = useLessonStore((s) => s.set);
  const complete = useUi((s) => s.completed);
  const setUi = useUi((s) => s.set);
  return (
    <div className="checkpoint" data-testid="checkpoint">
      <h4>✓ {t('checkpoint.title')}</h4>
      <div className="predict-q">{withMath(tr(q.question))}</div>
      <div className="predict-choices col">
        {q.choices.map((c) => (
          <button
            key={c.id}
            type="button"
            className={`btn sm ${picked === c.id ? (c.id === q.answer ? 'right' : 'wrong') : ''}`}
            aria-pressed={picked === c.id}
            onClick={() => {
              setStore(key, c.id);
              if (c.id === q.answer) setUi({ completed: { ...complete, [id]: true } });
            }}
          >
            {withMath(tr(c.label))}
          </button>
        ))}
      </div>
      {picked && (
        <div className="predict-explain" role="status">
          <strong>{picked === q.answer ? t('predict.right') : t('predict.wrong')}</strong> {withMath(tr(q.explain))}
        </div>
      )}
    </div>
  );
}

/** Observe → Predict → Experiment → Explain → Verify badge strip. */
export function Cycle({ active }: { active: 0 | 1 | 2 | 3 | 4 }) {
  const { t } = useT();
  const keys = ['cycle.observe', 'cycle.predict', 'cycle.experiment', 'cycle.explain', 'cycle.verify'] as const;
  return (
    <ol className="cycle" aria-label="learning cycle">
      {keys.map((k, i) => (
        <li key={k} className={i === active ? 'on' : i < active ? 'done' : ''}>
          {t(k)}
        </li>
      ))}
    </ol>
  );
}

/** Section heading inside a panel. */
export function H({ children }: { children: Bi }) {
  const { tr } = useT();
  return <h3>{tr(children)}</h3>;
}
