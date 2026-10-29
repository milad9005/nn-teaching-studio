// A horizontal number line with labelled markers and arrows from 0. Always drawn left-to-right,
// also in Persian, because numbers grow to the right in both languages.

import { useLayoutEffect, useRef, useState } from 'react';
import { fmt, niceTicks } from '../engine/numeric';
import { col, type Color } from './Plot2D';

export interface NumberMark {
  at: number;
  label?: string;
  color: Color;
  /** Draw an arrow from `from` (default 0) to this mark. */
  arrow?: boolean;
  from?: number;
}

export function NumberLine({
  domain,
  marks,
  height = 90,
  onDrag,
  testId,
}: {
  domain: [number, number];
  marks: NumberMark[];
  height?: number;
  /** Called with the number under the pointer while dragging. */
  onDrag?: (v: number) => void;
  testId?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(480);
  const [drag, setDrag] = useState(false);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(220, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const pad = 18;
  const [d0, d1] = domain;
  const sx = (v: number) => pad + ((v - d0) / (d1 - d0)) * (W - 2 * pad);
  const inv = (px: number) => d0 + ((px - pad) / (W - 2 * pad)) * (d1 - d0);
  const yAxis = height - 34;
  const ticks = niceTicks(d0, d1, Math.max(4, Math.floor(W / 45)));
  const pick = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
    onDrag?.(inv(e.clientX - r.left));
  };
  return (
    <div ref={ref} dir="ltr" className="numberline" data-testid={testId}>
      <svg
        width={W}
        height={height}
        role="img"
        aria-label="number line"
        style={{ cursor: onDrag ? (drag ? 'grabbing' : 'grab') : undefined, touchAction: 'none' }}
        onPointerDown={(e) => {
          if (!onDrag) return;
          (e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId);
          setDrag(true);
          pick(e);
        }}
        onPointerMove={(e) => drag && pick(e)}
        onPointerUp={() => setDrag(false)}
      >
        <defs>
          {marks.map((m, i) => (
            <marker key={i} id={`nl-arrow-${i}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill={col(m.color)} />
            </marker>
          ))}
        </defs>
        {d0 < 0 && <rect x={sx(d0)} y={yAxis - 6} width={sx(Math.min(0, d1)) - sx(d0)} height={12} fill="var(--c-L)" opacity={0.07} />}
        <line x1={pad} x2={W - pad} y1={yAxis} y2={yAxis} stroke="var(--line-strong)" strokeWidth={1.5} />
        {ticks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={yAxis - (t === 0 ? 8 : 4)} y2={yAxis + (t === 0 ? 8 : 4)} stroke={t === 0 ? 'var(--ink)' : 'var(--line-strong)'} strokeWidth={t === 0 ? 2 : 1} />
            <text x={sx(t)} y={yAxis + 22} textAnchor="middle" fontSize={11} fill={t === 0 ? 'var(--ink)' : 'var(--muted)'} fontWeight={t === 0 ? 700 : 400}>
              {fmt(t, 3)}
            </text>
          </g>
        ))}
        {marks.map((m, i) => {
          const y = yAxis - 16 - i * 14;
          const x = sx(Math.max(d0, Math.min(d1, m.at)));
          return (
            <g key={i}>
              {m.arrow && Math.abs(m.at - (m.from ?? 0)) > 1e-9 && (
                <line x1={sx(m.from ?? 0)} x2={x} y1={y} y2={y} stroke={col(m.color)} strokeWidth={2.5} markerEnd={`url(#nl-arrow-${i})`} />
              )}
              <circle cx={x} cy={yAxis} r={6} fill={col(m.color)} stroke="var(--surface)" strokeWidth={2} />
              {m.label && (
                <text x={x} y={y - 6} textAnchor="middle" fontSize={12} fontWeight={600} fill={col(m.color)}>
                  {m.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
