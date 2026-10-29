import { useEffect, useRef, useState } from 'react';
import { useUi } from '../state/ui';
import { useT } from '../i18n';
import { LESSONS } from '../lessons';

export function TopBar() {
  const { t, tr, lang } = useT();
  const ui = useUi();
  const [menu, setMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!menu) return;
    const close = (e: MouseEvent) => !menuRef.current?.contains(e.target as Node) && setMenu(false);
    window.addEventListener('mousedown', close);
    return () => window.removeEventListener('mousedown', close);
  }, [menu]);
  const lesson = LESSONS[ui.lessonIndex] ?? LESSONS[0];
  const nextTheme = ui.theme === 'system' ? 'light' : ui.theme === 'light' ? 'dark' : 'system';
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="22" height="22">
            <circle cx="5" cy="7" r="2.4" fill="var(--c-x)" />
            <circle cx="5" cy="17" r="2.4" fill="var(--c-x)" />
            <circle cx="19" cy="12" r="2.8" fill="var(--c-yhat)" />
            <path d="M7 7.5 L16.5 11.5 M7 16.5 L16.5 12.5" stroke="var(--c-w)" strokeWidth="1.8" />
          </svg>
        </span>
        <span className="brand-name">{t('appName')}</span>
      </div>
      <div className="seg mode-tabs" role="tablist">
        {(['guided', 'explore'] as const).map((m) => (
          <button key={m} role="tab" aria-selected={ui.mode === m} aria-pressed={ui.mode === m} onClick={() => ui.set({ mode: m, maximized: null })} data-testid={`mode-${m}`}>
            {t(`mode.${m}`)}
          </button>
        ))}
      </div>
      {ui.mode === 'guided' && (
        <div className="lesson-menu" ref={menuRef}>
          <button className="btn ghost" onClick={() => setMenu(!menu)} aria-expanded={menu} data-testid="lesson-menu">
            <span className="num">{String(ui.lessonIndex + 1).padStart(2, '0')}</span> {tr(lesson.short)} ▾
          </button>
          {menu && (
            <ol className="lesson-list" role="menu">
              {LESSONS.map((l, i) => (
                <li key={l.id}>
                  <button
                    role="menuitem"
                    className={i === ui.lessonIndex ? 'on' : ''}
                    onClick={() => {
                      ui.goLesson(i);
                      setMenu(false);
                    }}
                  >
                    <span className="num">{String(i + 1).padStart(2, '0')}</span>
                    <span className="grow">{tr(l.title)}</span>
                    {ui.completed[l.id] && <span className="c-pos">✓</span>}
                  </button>
                </li>
              ))}
            </ol>
          )}
        </div>
      )}
      <div className="grow" />
      <button className="btn ghost sm" onClick={() => ui.set({ lang: lang === 'en' ? 'fa' : 'en' })} data-testid="lang-toggle" lang={lang === 'en' ? 'fa' : 'en'}>
        {t('lang.switch')}
      </button>
      <button className="btn ghost sm" onClick={() => ui.set({ theme: nextTheme })} title={t('theme.toggle')} data-testid="theme-toggle">
        {ui.theme === 'dark' ? '☾' : ui.theme === 'light' ? '☀' : '◐'} {t(`theme.${ui.theme}`)}
      </button>
      <label className="small row reduce-motion" title={t('reducedMotion')}>
        <input type="checkbox" checked={ui.reducedMotion} onChange={() => ui.toggle('reducedMotion')} />
        {t('reducedMotion')}
      </label>
      <button className="btn primary sm" onClick={() => ui.set({ presentation: true, reveal: 0 })} data-testid="present">
        ▣ {t('mode.present')}
      </button>
    </header>
  );
}
