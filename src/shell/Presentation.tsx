// Presentation layer for recording or teaching: no menus, a big visual, keyboard control,
// optional captions, laser pointer and freehand annotations.

import { useEffect, useRef, useState } from 'react';
import { useUi } from '../state/ui';
import { useT } from '../i18n';
import { LESSONS } from '../story';
import { stepBy } from '../story/StoryView';

const typing = (e: KeyboardEvent) => {
  const el = e.target as HTMLElement;
  return el.tagName === 'INPUT' || el.tagName === 'SELECT' || el.tagName === 'TEXTAREA' || el.isContentEditable;
};

export function usePresentationKeys() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const ui = useUi.getState();
      if (!ui.presentation || typing(e) || e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key;
      // Steps (arrows, space) are handled by the lesson view itself.
      if (k === 'PageDown') {
        e.preventDefault();
        if (ui.lessonIndex < LESSONS.length - 1) ui.goLesson(ui.lessonIndex + 1);
      } else if (k === 'PageUp') {
        e.preventDefault();
        if (ui.lessonIndex > 0) ui.goLesson(ui.lessonIndex - 1);
      } else if (k === 't' || k === 'T') ui.toggle('captions');
      else if (k === 'l' || k === 'L') ui.set({ laser: !ui.laser, draw: false });
      else if (k === 'd' || k === 'D') ui.set({ draw: !ui.draw, laser: false });
      else if (k === 'c' || k === 'C') ui.set({ clearInk: ui.clearInk + 1 });
      else if (k === 'f' || k === 'F') toggleFullscreen();
      else if (k === 'Escape') ui.set({ presentation: false, laser: false, draw: false });
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}

export function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen?.();
  else document.documentElement.requestFullscreen?.().catch(() => undefined);
}

export function PresentationBar() {
  const { t } = useT();
  const ui = useUi();
  const lesson = LESSONS[ui.lessonIndex];
  return (
    <div className="present-bar" role="toolbar" data-testid="present-bar">
      {ui.mode === 'guided' && (
        <>
          <button className="btn sm" onClick={() => stepBy(-1)}>
            {t('nav.prev')}
          </button>
          <span className="num small">
            {ui.step + 1} / {lesson.steps.length}
          </span>
          <button className="btn sm primary" onClick={() => stepBy(1)} data-testid="present-next">
            {t('nav.next')}
          </button>
          <span className="sep" />
          <button className={`btn sm ${ui.captions ? 'active' : ''}`} onClick={() => ui.toggle('captions')} data-testid="captions-toggle">
            {t('present.captions')}
          </button>
        </>
      )}
      <button className={`btn sm ${ui.laser ? 'active' : ''}`} onClick={() => ui.set({ laser: !ui.laser, draw: false })}>
        ◉ {t('present.laser')}
      </button>
      <button className={`btn sm ${ui.draw ? 'active' : ''}`} onClick={() => ui.set({ draw: !ui.draw, laser: false })}>
        ✎ {t('present.draw')}
      </button>
      <button className="btn sm" onClick={() => ui.set({ clearInk: ui.clearInk + 1 })}>
        {t('present.clear')}
      </button>
      <button className="btn sm" onClick={toggleFullscreen}>
        ⛶ {t('present.fullscreen')}
      </button>
      <button className="btn sm" onClick={() => ui.set({ presentation: false, laser: false, draw: false })} data-testid="present-exit">
        {t('mode.exitPresent')}
      </button>
      <span className="keys small muted">{t('present.keys')}</span>
    </div>
  );
}

/** Laser pointer and ink canvas, layered over the whole app. */
export function InkLayer() {
  const laser = useUi((s) => s.laser);
  const draw = useUi((s) => s.draw);
  const clearInk = useUi((s) => s.clearInk);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const drawing = useRef(false);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const fit = () => {
      const img = c.getContext('2d')!.getImageData(0, 0, c.width, c.height);
      c.width = window.innerWidth * devicePixelRatio;
      c.height = window.innerHeight * devicePixelRatio;
      c.getContext('2d')!.putImageData(img, 0, 0);
    };
    fit();
    window.addEventListener('resize', fit);
    return () => window.removeEventListener('resize', fit);
  }, []);

  useEffect(() => {
    const c = canvas.current;
    c?.getContext('2d')!.clearRect(0, 0, c.width, c.height);
  }, [clearInk]);

  useEffect(() => {
    if (!laser) return setPos(null);
    const move = (e: PointerEvent) => setPos({ x: e.clientX, y: e.clientY });
    window.addEventListener('pointermove', move);
    return () => window.removeEventListener('pointermove', move);
  }, [laser]);

  const point = (e: React.PointerEvent) => [e.clientX * devicePixelRatio, e.clientY * devicePixelRatio] as const;
  return (
    <>
      <canvas
        ref={canvas}
        className="ink"
        style={{ pointerEvents: draw ? 'auto' : 'none', cursor: draw ? 'crosshair' : undefined }}
        onPointerDown={(e) => {
          drawing.current = true;
          const ctx = canvas.current!.getContext('2d')!;
          ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--c-L');
          ctx.lineWidth = 3.5 * devicePixelRatio;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';
          ctx.beginPath();
          ctx.moveTo(...point(e));
        }}
        onPointerMove={(e) => {
          if (!drawing.current) return;
          const ctx = canvas.current!.getContext('2d')!;
          ctx.lineTo(...point(e));
          ctx.stroke();
        }}
        onPointerUp={() => (drawing.current = false)}
      />
      {pos && <div className="laser" style={{ left: pos.x, top: pos.y }} aria-hidden="true" />}
    </>
  );
}
