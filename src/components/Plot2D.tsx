// A small, exact SVG plotter. Every curve is sampled from the real function it is given.

import { useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { fmt, niceTicks } from '../engine/numeric';

export type Color = string;
const SEM = new Set(['x', 'w', 'b', 'z', 'a', 'yhat', 'y', 'L', 'g', 'lr', 'wpos', 'wneg']);
export const col = (c: Color) => (SEM.has(c) ? `var(--c-${c})` : c);

type Pt = [number, number];

export type PlotLayer =
  | { kind: 'fn'; f: (x: number) => number; color: Color; width?: number; dash?: string; samples?: number; opacity?: number; domain?: [number, number] }
  | { kind: 'line'; p: Pt; slope: number; color: Color; width?: number; dash?: string; opacity?: number }
  | { kind: 'segment'; from: Pt; to: Pt; color: Color; width?: number; dash?: string; arrow?: boolean; label?: string; opacity?: number }
  | { kind: 'points'; pts: { x: number; y: number; color?: Color; r?: number; hollow?: boolean; label?: string }[]; color: Color }
  | { kind: 'vline'; at: number; color: Color; dash?: string; label?: string }
  | { kind: 'hline'; at: number; color: Color; dash?: string; label?: string }
  | { kind: 'band'; x0: number; x1: number; color: Color; opacity?: number; label?: string }
  | { kind: 'polyline'; pts: Pt[]; color: Color; width?: number; dash?: string; markers?: boolean; opacity?: number }
  | { kind: 'text'; x: number; y: number; text: string; color: Color; anchor?: 'start' | 'middle' | 'end'; dy?: number };

export interface Plot2DProps {
  xDomain: [number, number];
  yDomain: [number, number] | 'auto';
  layers: PlotLayer[];
  xLabel?: ReactNode;
  yLabel?: ReactNode;
  height?: number;
  /** Called while dragging (or clicking) inside the plot, in data coordinates. */
  onDrag?: (x: number, y: number) => void;
  /** Shows a grab cursor and a halo at this point. */
  handle?: { x: number; y: number; color: Color };
  legend?: { label: ReactNode; color: Color; dash?: boolean }[];
  ariaLabel?: string;
  testId?: string;
}

const PAD = { l: 52, r: 16, t: 24, b: 40 };

function autoY(layers: PlotLayer[], xd: [number, number]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  const take = (v: number) => {
    if (Number.isFinite(v)) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  };
  for (const L of layers) {
    if (L.kind === 'fn') for (let i = 0; i <= 80; i++) take(L.f(xd[0] + ((xd[1] - xd[0]) * i) / 80));
    if (L.kind === 'points') L.pts.forEach((p) => take(p.y));
    if (L.kind === 'polyline') L.pts.forEach((p) => take(p[1]));
    if (L.kind === 'hline') take(L.at);
  }
  if (!Number.isFinite(lo)) return [-1, 1];
  if (hi - lo < 1e-9) return [lo - 1, hi + 1];
  const pad = (hi - lo) * 0.1;
  return [lo - pad, hi + pad];
}

export function Plot2D({ xDomain, yDomain, layers, xLabel, yLabel, height = 260, onDrag, handle, legend, ariaLabel, testId }: Plot2DProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(480);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(220, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const H = height;
  const yd = useMemo(() => (yDomain === 'auto' ? autoY(layers, xDomain) : yDomain), [yDomain, layers, xDomain]);
  const [x0, x1] = xDomain;
  const [y0, y1] = yd;
  const sx = (x: number) => PAD.l + ((x - x0) / (x1 - x0)) * (W - PAD.l - PAD.r);
  const sy = (y: number) => PAD.t + (1 - (y - y0) / (y1 - y0)) * (H - PAD.t - PAD.b);
  const ix = (px: number) => x0 + ((px - PAD.l) / (W - PAD.l - PAD.r)) * (x1 - x0);
  const iy = (py: number) => y0 + (1 - (py - PAD.t) / (H - PAD.t - PAD.b)) * (y1 - y0);
  const clipY = (y: number) => Math.max(y0 - (y1 - y0) * 2, Math.min(y1 + (y1 - y0) * 2, y));
  const xt = niceTicks(x0, x1, Math.max(3, Math.floor(W / 80)));
  const yt = niceTicks(y0, y1, Math.max(3, Math.floor(H / 50)));
  const clipId = useMemo(() => 'clip' + Math.random().toString(36).slice(2), []);

  const dragging = useRef(false);
  const toData = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
    return [ix(((e.clientX - r.left) / r.width) * W), iy(((e.clientY - r.top) / r.height) * H)] as const;
  };

  const fnPath = (f: (x: number) => number, n: number, dom?: [number, number]) => {
    const a = Math.max(x0, dom?.[0] ?? x0);
    const b = Math.min(x1, dom?.[1] ?? x1);
    let d = '';
    let pen = false;
    for (let i = 0; i <= n; i++) {
      const x = a + ((b - a) * i) / n;
      const y = f(x);
      if (!Number.isFinite(y)) {
        pen = false;
        continue;
      }
      d += `${pen ? 'L' : 'M'}${sx(x).toFixed(2)},${sy(clipY(y)).toFixed(2)}`;
      pen = true;
    }
    return d;
  };

  return (
    <div ref={ref} className="plot2d" dir="ltr" data-testid={testId}>
      {legend && (
        <div className="plot-legend">
          {legend.map((l, i) => (
            <span key={i}>
              <i style={{ borderColor: col(l.color), borderStyle: l.dash ? 'dashed' : 'solid' }} />
              {l.label}
            </span>
          ))}
        </div>
      )}
      <svg
        width="100%"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={ariaLabel}
        style={{ cursor: onDrag ? 'crosshair' : undefined, touchAction: onDrag ? 'none' : undefined }}
        onPointerDown={(e) => {
          if (!onDrag) return;
          dragging.current = true;
          (e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId);
          onDrag(...toData(e));
        }}
        onPointerMove={(e) => onDrag && dragging.current && onDrag(...toData(e))}
        onPointerUp={() => (dragging.current = false)}
      >
        <defs>
          <clipPath id={clipId}>
            <rect x={PAD.l} y={PAD.t} width={W - PAD.l - PAD.r} height={H - PAD.t - PAD.b} />
          </clipPath>
          <marker id={clipId + 'arr'} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="context-stroke" />
          </marker>
        </defs>
        <g className="grid">
          {xt.map((t) => (
            <line key={'x' + t} x1={sx(t)} x2={sx(t)} y1={PAD.t} y2={H - PAD.b} />
          ))}
          {yt.map((t) => (
            <line key={'y' + t} x1={PAD.l} x2={W - PAD.r} y1={sy(t)} y2={sy(t)} />
          ))}
        </g>
        {y0 < 0 && y1 > 0 && <line className="axis0" x1={PAD.l} x2={W - PAD.r} y1={sy(0)} y2={sy(0)} />}
        {x0 < 0 && x1 > 0 && <line className="axis0" x1={sx(0)} x2={sx(0)} y1={PAD.t} y2={H - PAD.b} />}
        <rect className="frame" x={PAD.l} y={PAD.t} width={W - PAD.l - PAD.r} height={H - PAD.t - PAD.b} />
        <g className="ticks">
          {xt.map((t) => (
            <text key={'xt' + t} x={sx(t)} y={H - PAD.b + 16} textAnchor="middle">
              {fmt(t, 2)}
            </text>
          ))}
          {yt.map((t) => (
            <text key={'yt' + t} x={PAD.l - 6} y={sy(t) + 4} textAnchor="end">
              {fmt(t, 2)}
            </text>
          ))}
        </g>
        <g clipPath={`url(#${clipId})`}>
          {layers.map((L, k) => {
            switch (L.kind) {
              case 'band':
                return (
                  <g key={k}>
                    <rect x={sx(Math.max(x0, L.x0))} width={Math.max(0, sx(Math.min(x1, L.x1)) - sx(Math.max(x0, L.x0)))} y={PAD.t} height={H - PAD.t - PAD.b} fill={col(L.color)} opacity={L.opacity ?? 0.1} />
                    {L.label && (
                      <text x={(sx(Math.max(x0, L.x0)) + sx(Math.min(x1, L.x1))) / 2} y={PAD.t + 14} textAnchor="middle" className="lbl" fill={col(L.color)}>
                        {L.label}
                      </text>
                    )}
                  </g>
                );
              case 'fn':
                return <path key={k} d={fnPath(L.f, L.samples ?? Math.min(600, Math.max(120, W)), L.domain)} fill="none" stroke={col(L.color)} strokeWidth={L.width ?? 2.2} strokeDasharray={L.dash} opacity={L.opacity} strokeLinejoin="round" />;
              case 'line': {
                const f = (x: number) => L.p[1] + L.slope * (x - L.p[0]);
                return <path key={k} d={fnPath(f, 2)} fill="none" stroke={col(L.color)} strokeWidth={L.width ?? 1.6} strokeDasharray={L.dash} opacity={L.opacity} />;
              }
              case 'segment':
                return (
                  <g key={k}>
                    <line x1={sx(L.from[0])} y1={sy(clipY(L.from[1]))} x2={sx(L.to[0])} y2={sy(clipY(L.to[1]))} stroke={col(L.color)} strokeWidth={L.width ?? 1.6} strokeDasharray={L.dash} opacity={L.opacity} markerEnd={L.arrow ? `url(#${clipId}arr)` : undefined} />
                    {L.label && (
                      <text x={(sx(L.from[0]) + sx(L.to[0])) / 2 + 6} y={(sy(L.from[1]) + sy(L.to[1])) / 2 - 4} className="lbl" fill={col(L.color)}>
                        {L.label}
                      </text>
                    )}
                  </g>
                );
              case 'polyline':
                return (
                  <g key={k} opacity={L.opacity}>
                    <polyline points={L.pts.map((p) => `${sx(p[0])},${sy(clipY(p[1]))}`).join(' ')} fill="none" stroke={col(L.color)} strokeWidth={L.width ?? 1.8} strokeDasharray={L.dash} strokeLinejoin="round" />
                    {L.markers && L.pts.map((p, i) => <circle key={i} cx={sx(p[0])} cy={sy(clipY(p[1]))} r={2.6} fill={col(L.color)} />)}
                  </g>
                );
              case 'vline':
                return (
                  <g key={k}>
                    <line x1={sx(L.at)} x2={sx(L.at)} y1={PAD.t} y2={H - PAD.b} stroke={col(L.color)} strokeDasharray={L.dash ?? '4 4'} />
                    {L.label && (
                      <text x={sx(L.at) + 4} y={PAD.t + 12} className="lbl" fill={col(L.color)}>
                        {L.label}
                      </text>
                    )}
                  </g>
                );
              case 'hline':
                return (
                  <g key={k}>
                    <line x1={PAD.l} x2={W - PAD.r} y1={sy(L.at)} y2={sy(L.at)} stroke={col(L.color)} strokeDasharray={L.dash ?? '4 4'} />
                    {L.label && (
                      <text x={W - PAD.r - 4} y={sy(L.at) - 5} textAnchor="end" className="lbl" fill={col(L.color)}>
                        {L.label}
                      </text>
                    )}
                  </g>
                );
              case 'points':
                return (
                  <g key={k}>
                    {L.pts.map((p, i) =>
                      Number.isFinite(p.y) ? (
                        <g key={i}>
                          <circle cx={sx(p.x)} cy={sy(clipY(p.y))} r={p.r ?? 5} fill={p.hollow ? 'var(--surface)' : col(p.color ?? L.color)} stroke={col(p.color ?? L.color)} strokeWidth={p.hollow ? 2 : 1.5} />
                          {p.label && (
                            <text x={sx(p.x) + 8} y={sy(clipY(p.y)) - 8} className="lbl" fill={col(p.color ?? L.color)}>
                              {p.label}
                            </text>
                          )}
                        </g>
                      ) : null,
                    )}
                  </g>
                );
              case 'text':
                return (
                  <text key={k} x={sx(L.x)} y={sy(L.y) + (L.dy ?? 0)} textAnchor={L.anchor ?? 'start'} className="lbl" fill={col(L.color)}>
                    {L.text}
                  </text>
                );
            }
          })}
          {handle && Number.isFinite(handle.y) && (
            <g>
              <circle cx={sx(handle.x)} cy={sy(clipY(handle.y))} r={13} fill={col(handle.color)} opacity={0.15} />
              <circle cx={sx(handle.x)} cy={sy(clipY(handle.y))} r={6.5} fill={col(handle.color)} stroke="var(--surface)" strokeWidth={2} />
            </g>
          )}
        </g>
        {xLabel && (
          <foreignObject x={PAD.l} y={H - 22} width={W - PAD.l - PAD.r} height={22}>
            <div className="axis-label x">{xLabel}</div>
          </foreignObject>
        )}
        {yLabel && (
          <foreignObject x={0} y={0} width={PAD.l + 200} height={18}>
            <div className="axis-label y">{yLabel}</div>
          </foreignObject>
        )}
      </svg>
    </div>
  );
}
