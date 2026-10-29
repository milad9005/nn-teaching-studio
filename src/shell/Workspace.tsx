// Four-panel workspace: left (concept), center (visual), right (math), bottom (graphs).
// Panels resize by dragging the gutters, collapse, and the center or right panel can be
// maximized. Built on CSS grid so RTL mirrors the layout automatically.

import { useRef, type ReactNode } from 'react';
import { useUi, type PanelId } from '../state/ui';
import { useT, type StringKey } from '../i18n';

export function Panel({
  id,
  title,
  children,
  actions,
  maximizable,
}: {
  id: 'left' | 'center' | 'right' | 'bottom';
  title: StringKey;
  children: ReactNode;
  actions?: ReactNode;
  maximizable?: boolean;
}) {
  const { t } = useT();
  const togglePanel = useUi((s) => s.togglePanel);
  const maximized = useUi((s) => s.maximized);
  const set = useUi((s) => s.set);
  const isMax = maximized === id;
  return (
    <section className={`panel panel-${id}`} aria-label={t(title)} data-testid={`panel-${id}`}>
      <header className="panel-head">
        <h2>{t(title)}</h2>
        <div className="row panel-actions">
          {actions}
          {maximizable && (
            <button className="btn ghost sm icon" title={isMax ? t('panel.restore') : t('panel.maximize')} aria-label={isMax ? t('panel.restore') : t('panel.maximize')} onClick={() => set({ maximized: isMax ? null : (id as 'center' | 'right') })} data-testid={`max-${id}`}>
              {isMax ? '⤡' : '⤢'}
            </button>
          )}
          {id !== 'center' && !maximized && (
            <button className="btn ghost sm icon" title={t('panel.collapse')} aria-label={t('panel.collapse')} onClick={() => togglePanel(id as PanelId)} data-testid={`collapse-${id}`}>
              ×
            </button>
          )}
        </div>
      </header>
      <div className="panel-body">{children}</div>
    </section>
  );
}

function Gutter({ panel, axis }: { panel: PanelId; axis: 'x' | 'y' }) {
  const set = useUi((s) => s.set);
  const start = useRef<{ p: number; size: number } | null>(null);
  const { dir } = useT();
  return (
    <div
      className={`gutter gutter-${axis}`}
      role="separator"
      aria-orientation={axis === 'x' ? 'vertical' : 'horizontal'}
      tabIndex={0}
      onKeyDown={(e) => {
        const size = useUi.getState().size;
        const d = e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -20 : e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 20 : 0;
        if (!d) return;
        const sign = panel === 'left' ? 1 : -1;
        const flip = axis === 'x' && dir === 'rtl' ? -1 : 1;
        set({ size: { ...size, [panel]: clampSize(panel, size[panel] + d * sign * flip) } });
      }}
      onPointerDown={(e) => {
        (e.target as HTMLElement).setPointerCapture(e.pointerId);
        start.current = { p: axis === 'x' ? e.clientX : e.clientY, size: useUi.getState().size[panel] };
      }}
      onPointerMove={(e) => {
        if (!start.current) return;
        const delta = (axis === 'x' ? e.clientX : e.clientY) - start.current.p;
        const sign = panel === 'left' ? 1 : -1;
        const flip = axis === 'x' && dir === 'rtl' ? -1 : 1;
        const size = useUi.getState().size;
        set({ size: { ...size, [panel]: clampSize(panel, start.current.size + delta * sign * flip) } });
      }}
      onPointerUp={() => (start.current = null)}
    />
  );
}

const clampSize = (p: PanelId, v: number) => Math.round(Math.max(p === 'bottom' ? 120 : 240, Math.min(p === 'bottom' ? 640 : 760, v)));

function Reopen({ panel }: { panel: PanelId }) {
  const { t } = useT();
  const togglePanel = useUi((s) => s.togglePanel);
  const label = t(panel === 'left' ? 'panel.explain' : panel === 'right' ? 'panel.math' : 'panel.graphs');
  return (
    <button className={`reopen reopen-${panel}`} onClick={() => togglePanel(panel)} title={`${t('panel.expand')}: ${label}`} data-testid={`reopen-${panel}`}>
      {label}
    </button>
  );
}

export function Workspace({ left, center, right, bottom }: { left: ReactNode; center: ReactNode; right?: ReactNode; bottom?: ReactNode }) {
  const collapsed = useUi((s) => s.collapsed);
  const size = useUi((s) => s.size);
  const maximized = useUi((s) => s.maximized);
  const showLeft = !collapsed.left && !maximized;
  const showRight = !!right && !collapsed.right && maximized !== 'center';
  const showBottom = !!bottom && !collapsed.bottom && !maximized;
  const showCenter = maximized !== 'right';
  const cols = [showLeft ? `${size.left}px 6px` : '', showCenter ? 'minmax(320px, 1fr)' : '', showRight ? (showCenter ? `6px ${size.right}px` : '1fr') : ''].filter(Boolean).join(' ');
  return (
    <div className="workspace" style={{ gridTemplateColumns: cols }} data-maximized={maximized ?? 'none'}>
      {showLeft && (
        <>
          <div className="ws-col">{left}</div>
          <Gutter panel="left" axis="x" />
        </>
      )}
      {showCenter && (
        <div className="ws-col ws-center" style={{ gridTemplateRows: showBottom ? `1fr 6px ${size.bottom}px` : '1fr' }}>
          {center}
          {showBottom && (
            <>
              <Gutter panel="bottom" axis="y" />
              {bottom}
            </>
          )}
          <div className="reopen-bar">
            {!showLeft && !maximized && <Reopen panel="left" />}
            {!!bottom && !showBottom && !maximized && <Reopen panel="bottom" />}
            {!!right && !showRight && !maximized && <Reopen panel="right" />}
          </div>
        </div>
      )}
      {showRight && (
        <>
          {showCenter && <Gutter panel="right" axis="x" />}
          <div className="ws-col">{right}</div>
        </>
      )}
    </div>
  );
}
