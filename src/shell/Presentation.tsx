// Presentation layer: keyboard control, step reveal, laser pointer and freehand annotations.

import { useEffect, useRef, useState } from 'react';
import { useUi } from '../state/ui';
import { useT } from '../i18n';
import { LESSONS } from '../lessons';
import { useLab } from '../state/lab';

const typing = (e: KeyboardEvent) => {
  const el = e.target as HTMLElement;
  return el.tagName === 'INPUT' || el.tagName === 'SELECT' || el.tagName === 'TEXTAREA' || el.isContentEditable;
};

export function usePresentationKeys() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const ui = useUi.getState();
      if (!ui.presentation || typing(e) || e.metaKey || e.ctrlKey || e.altKey) return;
      const lesson = LESSONS[ui.lessonIndex];
      const maxReveal = ui.mode === 'guided' ? lesson.revealSteps : 0;
      const k = e.key;
      if (k === 'ArrowRight' || k === ' ') {
        e.preventDefault();
        // With RTL the visual direction flips but "next" stays on the forward key.
        if (ui.reveal < maxReveal) ui.set({ reveal: ui.reveal + 1 });
        else useLab.getState().forwardStep();
      } else if (k === 'ArrowLeft') {
        e.preventDefault();
        if (ui.reveal > 0) ui.set({ reveal: ui.reveal - 1 });
      } else if (k === 'PageDown') {
        e.preventDefault();
        if (ui.lessonIndex < LESSONS.length - 1) ui.set({ lessonIndex: ui.lessonIndex + 1, reveal: 0, mode: 'guided' });
      } else if (k === 'PageUp') {
        e.preventDefault();
        if (ui.lessonIndex > 0) ui.set({ lessonIndex: ui.lessonIndex - 1, reveal: 0, mode: 'guided' });
      } else if (k === 'a' || k === 'A') ui.toggle('hideAdvanced');
      else if (k === 'l' || k === 'L') ui.set({ laser: !ui.laser, draw: false });
      else if (k === 'd' || k === 'D') ui.set({ draw: !ui.draw, laser: false });
      else if (k === 'c' || k === 'C') ui.set({ clearInk: ui.clearInk + 1 });
      else if (k === 'f' || k === 'F') toggleFullscreen();
      else if (k === 'b' || k === 'B') useLab.getState().backwardStep();
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
  const maxReveal = ui.mode === 'guided' ? lesson.revealSteps : 0;
  return (
    <div className="present-bar" role="toolbar" data-testid="present-bar">
      <button className="btn sm" onClick={() => ui.set({ reveal: Math.max(0, ui.reveal - 1) })} disabled={ui.reveal === 0}>
        ←
      </button>
      <span className="num small">
        {Math.min(ui.reveal, maxReveal)} / {maxReveal}
      </span>
      <button className="btn sm primary" onClick={() => ui.set({ reveal: Math.min(maxReveal, ui.reveal + 1) })} disabled={ui.reveal >= maxReveal} data-testid="present-next">
        {t('present.next')} →
      </button>
      <span className="sep" />
      <button className={`btn sm ${ui.laser ? 'active' : ''}`} onClick={() => ui.set({ laser: !ui.laser, draw: false })}>
        ◉ {t('present.laser')}
      </button>
      <button className={`btn sm ${ui.draw ? 'active' : ''}`} onClick={() => ui.set({ draw: !ui.draw, laser: false })}>
        ✎ {t('present.draw')}
      </button>
      <button className="btn sm" onClick={() => ui.set({ clearInk: ui.clearInk + 1 })}>
        {t('present.clear')}
      </button>
      <button className={`btn sm ${ui.hideAdvanced ? 'active' : ''}`} onClick={() => ui.toggle('hideAdvanced')}>
        {t('present.hideMath')}
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
