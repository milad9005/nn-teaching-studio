// Lesson 7 · Many knobs: the gradient. One derivative per knob; the list of them is the gradient.
// Script: docs/storyboard.md, lesson 7. L(w, b) = (2w + b − 10)², start (3, 0): L = 16, ∇L = (−16, −8).

import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import type { Data, Layout } from 'plotly.js';
import { bi, useT } from '../../i18n';
import { useUi } from '../../state/ui';
import { Tex } from '../../components/Tex';
import { cssVar, useThemeKey } from '../../components/PlotlyChart';
import { fmt, linspace, tex } from '../../engine';
import type { StoryLesson } from '../types';
import { Appear, C, Knob, Machine, Meter, Note, ValueCard } from '../kit';
import { useTween, useTweenArray, useTicker } from '../useTween';
import { dLdb, dLdw, err, levelSegment, loss2, path2, START, X, type Box, type Pt } from './math-gradient';

const ID = 'gradient';
const ETA = 0.05;
const METER_MAX = 100; // the same range as lessons 4–5
const [W0, B0] = START;
const G0: Pt = [dLdw(W0, B0), dLdb(W0, B0)];
const L0 = loss2(W0, B0);
const DESC_N = 8;
const DESC = path2(START, ETA, DESC_N);
/** The gradient arrow on the map is drawn at this fraction of its length. */
const ARROW = 0.1;
/** Persian digits for numbers inside Persian prose (not inside math). */
const fa = (v: number | string) => String(v).replace(/[0-9]/g, (d) => '۰۱۲۳۴۵۶۷۸۹'[Number(d)]).replace('.', '٫');

// ---------- the map ----------
const BOX: Box = { w0: 0, w1: 9, b0: -3, b1: 4 };
const MP = { l: 50, r: 14, t: 14, b: 46 };
const KPX = 60; // pixels per unit, the same on both axes
const MW = MP.l + MP.r + (BOX.w1 - BOX.w0) * KPX;
const MH = MP.t + MP.b + (BOX.b1 - BOX.b0) * KPX;
const mx = (w: number) => MP.l + (w - BOX.w0) * KPX;
const my = (b: number) => MP.t + (BOX.b1 - b) * KPX;
/** Height lines at e = 2w + b − 10 = ±2, ±4, …: L = 4, 16, 36, 64, 100, 144. */
const LEVELS = [2, 4, 6, 8, 10, 12].flatMap((e) => [-e, e]);
const E_MAX = 14;

function MapBackground() {
  // L only depends on e, so the colour changes only across the valley: a linear gradient.
  const O: Pt = [mx(5), my(0)]; // e = 0 here
  const u: Pt = [2 / Math.sqrt(5), -1 / Math.sqrt(5)]; // screen direction in which e grows
  const pxPerE = KPX / Math.sqrt(5);
  const at = (e: number): Pt => [O[0] + u[0] * e * pxPerE, O[1] + u[1] * e * pxPerE];
  const [x1, y1] = at(-E_MAX);
  const [x2, y2] = at(E_MAX);
  const stops = linspace(-E_MAX, E_MAX, 29);
  return (
    <>
      <defs>
        <linearGradient id="gr-bg" gradientUnits="userSpaceOnUse" x1={x1} y1={y1} x2={x2} y2={y2}>
          {stops.map((e) => (
            <stop key={e} offset={(e + E_MAX) / (2 * E_MAX)} stopColor={C.loss} stopOpacity={0.5 * (Math.abs(e) / E_MAX) ** 0.9} />
          ))}
        </linearGradient>
        <clipPath id="gr-clip">
          <rect x={mx(BOX.w0)} y={my(BOX.b1)} width={(BOX.w1 - BOX.w0) * KPX} height={(BOX.b1 - BOX.b0) * KPX} />
        </clipPath>
        <marker id="gr-arr" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="context-stroke" />
        </marker>
      </defs>
      <rect x={mx(BOX.w0)} y={my(BOX.b1)} width={(BOX.w1 - BOX.w0) * KPX} height={(BOX.b1 - BOX.b0) * KPX} fill="var(--surface)" />
      <rect x={mx(BOX.w0)} y={my(BOX.b1)} width={(BOX.w1 - BOX.w0) * KPX} height={(BOX.b1 - BOX.b0) * KPX} fill="url(#gr-bg)" />
    </>
  );
}

function HeightLines({ through }: { through: number }) {
  return (
    <g>
      {LEVELS.map((e) => {
        const s = levelSegment(e, BOX);
        if (!s) return null;
        const [a, b] = s;
        const bold = Math.abs(e - through) < 1e-9;
        // Label where the line enters the map from the top or the left.
        const lab = a[1] > b[1] ? a : b;
        return (
          <g key={e}>
            <line x1={mx(a[0])} y1={my(a[1])} x2={mx(b[0])} y2={my(b[1])} stroke={C.loss} strokeWidth={bold ? 3 : 1.6} opacity={bold ? 0.95 : 0.55} />
            <text x={mx(lab[0]) + (lab[1] >= BOX.b1 - 1e-9 ? 6 : 8)} y={my(lab[1]) + (lab[1] >= BOX.b1 - 1e-9 ? 16 : -6)} fontSize={14} fontWeight={bold ? 800 : 600} fill={C.loss} className="halo">
              {fmt(e * e, 0)}
            </text>
          </g>
        );
      })}
      {(() => {
        const s = levelSegment(0, BOX)!;
        return <line x1={mx(s[0][0])} y1={my(s[0][1])} x2={mx(s[1][0])} y2={my(s[1][1])} stroke={C.output} strokeWidth={2.5} strokeDasharray="7 6" />;
      })()}
    </g>
  );
}

interface MapProps {
  pos: Pt;
  trail: Pt[];
  slice: 'w' | 'b' | null;
  arrow: boolean;
}

function ContourMap({ pos, trail, slice, arrow }: MapProps) {
  const { tr } = useT();
  const [w, b] = useTweenArray(pos, 550);
  const sliceK = useTween(slice ? 1 : 0, 500);
  const arrowK = useTween(arrow ? 1 : 0, 700);
  const g = [dLdw(w, b), dLdb(w, b)];
  const floor = levelSegment(0, BOX)!;
  return (
    <svg viewBox={`0 0 ${MW} ${MH}`} width="100%" style={{ display: 'block', maxWidth: 640, direction: 'ltr' }} role="img" aria-label="top-view map of the loss">
      <style>{'.halo{paint-order:stroke;stroke:var(--surface);stroke-width:4px;stroke-linejoin:round}'}</style>
      <MapBackground />
      <g clipPath="url(#gr-clip)">
        <HeightLines through={err(W0, B0)} />
        {/* the slices: a horizontal line (only w moves) or a vertical one (only b moves) */}
        {slice === 'w' && <line x1={mx(BOX.w0)} x2={mx(BOX.w1)} y1={my(b)} y2={my(b)} stroke={C.weight} strokeWidth={5} opacity={0.85 * sliceK} />}
        {slice === 'b' && <line x1={mx(w)} x2={mx(w)} y1={my(BOX.b0)} y2={my(BOX.b1)} stroke={C.bias} strokeWidth={5} opacity={0.85 * sliceK} />}
        {trail.length > 1 && <polyline points={trail.map(([p, q]) => `${mx(p)},${my(q)}`).join(' ')} fill="none" stroke={C.weight} strokeWidth={2.5} strokeDasharray="5 5" />}
        {trail.slice(0, -1).map(([p, q], i) => (
          <circle key={i} cx={mx(p)} cy={my(q)} r={5} fill="var(--surface)" stroke={C.weight} strokeWidth={2.5} />
        ))}
        {arrowK > 0.01 && (
          <g opacity={arrowK}>
            <line x1={mx(w)} y1={my(b)} x2={mx(w + ARROW * g[0] * arrowK)} y2={my(b + ARROW * g[1] * arrowK)} stroke={C.slope} strokeWidth={5} markerEnd="url(#gr-arr)" />
            <text x={mx(w + ARROW * g[0]) + 10} y={my(b + ARROW * g[1]) + 40} textAnchor="middle" fontSize={20} fontWeight={800} fill={C.slope} className="halo">
              {`∇L ${tr(bi('uphill', 'سربالایی'))}`}
            </text>
          </g>
        )}
        <circle cx={mx(w)} cy={my(b)} r={10} fill={C.weight} stroke="var(--surface)" strokeWidth={3} />
      </g>
      <text x={mx((floor[0][0] + floor[1][0]) / 2) + 10} y={my((floor[0][1] + floor[1][1]) / 2) + 4} fontSize={14} fontWeight={700} fill={C.output} className="halo" textAnchor="start">
        L = 0
      </text>
      <rect x={mx(BOX.w0)} y={my(BOX.b1)} width={(BOX.w1 - BOX.w0) * KPX} height={(BOX.b1 - BOX.b0) * KPX} fill="none" stroke="var(--line-strong)" />
      {linspace(BOX.w0, BOX.w1, BOX.w1 - BOX.w0 + 1).map((v) => (
        <text key={'w' + v} x={mx(v)} y={my(BOX.b0) + 20} textAnchor="middle" fontSize={14} fill="var(--muted)">
          {v}
        </text>
      ))}
      {linspace(BOX.b0, BOX.b1, BOX.b1 - BOX.b0 + 1).map((v) => (
        <text key={'b' + v} x={mx(BOX.w0) - 8} y={my(v) + 5} textAnchor="end" fontSize={14} fill="var(--muted)">
          {fmt(v, 0)}
        </text>
      ))}
      <text x={mx(BOX.w1)} y={MH - 4} textAnchor="end" fontSize={18} fontWeight={700} fill={C.weight}>
        w →
      </text>
      <text x={14} y={my(BOX.b1) + 4} textAnchor="middle" fontSize={18} fontWeight={700} fill={C.bias}>
        b
      </text>
      <text x={14} y={my(BOX.b1) + 22} textAnchor="middle" fontSize={16} fontWeight={700} fill={C.bias}>
        ↑
      </text>
    </svg>
  );
}

// ---------- the 3-D landscape (Plotly, loaded lazily) ----------
type PlotlyModule = typeof import('plotly.js-dist-min');
let plotlyPromise: Promise<PlotlyModule> | null = null;
const loadPlotly = () => (plotlyPromise ??= import('plotly.js-dist-min').then((m) => (m as unknown as { default: PlotlyModule }).default ?? m));

const R = 2.2;
const deg = Math.PI / 180;
const eye = (theta: number, phi: number) => ({ x: R * Math.cos(phi) * Math.cos(theta), y: R * Math.cos(phi) * Math.sin(theta), z: R * Math.sin(phi) });
const VIEW0 = { theta: -62 * deg, phi: 26 * deg };
const TOP = { theta: -90 * deg, phi: 89.8 * deg };
const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

/** The landscape. `mode` "turn": slowly swings around; "top": the camera tilts to look straight down. */
function Landscape({ mode, height }: { mode: 'turn' | 'top'; height: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const theme = useThemeKey();
  const reduced = useUi((s) => s.reducedMotion);
  const [ready, setReady] = useState(false);
  const view = useRef({ ...VIEW0 });

  const data = useMemo<Data[]>(() => {
    const ws = linspace(BOX.w0, BOX.w1, 46);
    const bs = linspace(BOX.b0, BOX.b1, 36);
    const loss = cssVar('--c-L');
    const lines: Data[] = LEVELS.flatMap((e) => {
      const s = levelSegment(e, BOX);
      if (!s) return [];
      const pts = linspace(0, 1, 12).map((t) => [s[0][0] + (s[1][0] - s[0][0]) * t, s[0][1] + (s[1][1] - s[0][1]) * t]);
      return [{ type: 'scatter3d', mode: 'lines', x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), z: pts.map(() => e * e + 0.6), line: { color: loss, width: e === err(W0, B0) ? 7 : 3 }, hoverinfo: 'skip' } as Data];
    });
    return [
      {
        type: 'surface',
        x: ws,
        y: bs,
        z: bs.map((b) => ws.map((w) => loss2(w, b))),
        surfacecolor: bs.map((b) => ws.map((w) => Math.abs(err(w, b)))),
        cmin: 0,
        cmax: E_MAX,
        colorscale: [
          [0, cssVar('--surface')],
          [1, loss],
        ],
        opacity: 0.92,
        showscale: false,
        hoverinfo: 'skip',
        lighting: { ambient: 0.85, diffuse: 0.4, specular: 0.05 },
      } as Data,
      ...lines,
      {
        type: 'scatter3d',
        mode: 'markers+text',
        x: [W0],
        y: [B0],
        z: [L0 + 6],
        text: [`  (${W0}, ${B0}, ${L0})`],
        textposition: 'middle right',
        textfont: { size: 16, color: cssVar('--c-w') },
        marker: { size: 9, color: cssVar('--c-w'), line: { color: cssVar('--surface'), width: 2 } },
        hoverinfo: 'skip',
      } as unknown as Data,
    ];
    // theme: re-resolve CSS colours when the theme changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme]);

  const layout = useMemo<Partial<Layout>>(() => {
    const ink = cssVar('--ink-2');
    const ax = (t: string, color: string) => ({ title: { text: t, font: { size: 18, color } }, color: ink, gridcolor: cssVar('--line'), showbackground: false, tickfont: { size: 12 } });
    return {
      height,
      margin: { l: 0, r: 0, t: 0, b: 0 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      showlegend: false,
      font: { family: 'IBM Plex Sans, system-ui, sans-serif', color: ink },
      scene: {
        xaxis: { ...ax('w', cssVar('--c-w')), range: [BOX.w0, BOX.w1] },
        yaxis: { ...ax('b', cssVar('--c-b')), range: [BOX.b0, BOX.b1] },
        zaxis: { ...ax('L', cssVar('--c-L')), range: [0, 230] },
        aspectmode: 'manual',
        aspectratio: { x: 1.3 * ((BOX.w1 - BOX.w0) / (BOX.b1 - BOX.b0)), y: 1.3, z: 0.75 },
        camera: { eye: eye(view.current.theta, view.current.phi), up: { x: 0, y: 0, z: 1 }, center: { x: 0, y: 0, z: -0.15 }, projection: { type: 'orthographic' } },
      },
    } as Partial<Layout>;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme, height]);

  useLayoutEffect(() => {
    let alive = true;
    const el = ref.current;
    loadPlotly().then((Plotly) => {
      if (!alive || !el) return;
      Plotly.react(el, data, layout, { displaylogo: false, displayModeBar: false, scrollZoom: false, responsive: true });
      setReady(true);
    });
    return () => {
      alive = false;
    };
  }, [data, layout]);

  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) loadPlotly().then((P) => P.purge(el));
    };
  }, []);

  // Camera motion.
  useEffect(() => {
    if (!ready) return;
    const el = ref.current;
    if (!el) return;
    let raf = 0;
    let busy = false;
    let last = 0;
    const start = performance.now();
    const from = { ...view.current };
    const apply = (theta: number, phi: number) => {
      view.current = { theta, phi };
      if (busy) return;
      busy = true;
      loadPlotly()
        .then((P) => P.relayout(el, { 'scene.camera.eye': eye(theta, phi) } as Partial<Layout>))
        .finally(() => (busy = false));
    };
    if (reduced) {
      const v = mode === 'top' ? TOP : VIEW0;
      apply(v.theta, v.phi);
      return;
    }
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      if (now - last < 33) return;
      last = now;
      const s = (now - start) / 1000;
      if (mode === 'turn') {
        // Ease in from wherever the camera is, then swing slowly left and right.
        const k = Math.min(1, s / 1.2);
        const theta = VIEW0.theta + 22 * deg * Math.sin(s * 0.45);
        const phi = VIEW0.phi;
        apply(from.theta + (theta - from.theta) * ease(k), from.phi + (phi - from.phi) * ease(k));
      } else {
        const k = ease(Math.min(1, s / 1.8));
        apply(from.theta + (TOP.theta - from.theta) * k, from.phi + (TOP.phi - from.phi) * k);
        if (k >= 1) cancelAnimationFrame(raf);
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [ready, mode, reduced]);

  return (
    <div dir="ltr" style={{ position: 'relative', height, pointerEvents: 'none' }}>
      {!ready && <div className="muted small" style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center' }}>…</div>}
      <div ref={ref} />
    </div>
  );
}

// ---------- a slice, lifted out of the map as a U curve with its tangent ----------
const SW = 440;
const SH = 200;
const SP = { l: 44, r: 16, t: 16, b: 40 };

function Slice({ which, show }: { which: 'w' | 'b'; show: boolean }) {
  const { tr } = useT();
  const k = useTween(show ? 1 : 0, 1100);
  const isW = which === 'w';
  const dom: [number, number] = isW ? [0, 10] : [-4, 12];
  const yMax = isW ? 100 : 64;
  const f = (v: number) => (isW ? loss2(v, B0) : loss2(W0, v));
  const at = isW ? W0 : B0;
  const sl = isW ? G0[0] : G0[1];
  const color = isW ? C.weight : C.bias;
  const px = (v: number) => SP.l + ((v - dom[0]) / (dom[1] - dom[0])) * (SW - SP.l - SP.r);
  const py = (L: number) => SH - SP.b - (k * L * (SH - SP.t - SP.b)) / yMax;
  let d = '';
  for (let i = 0; i <= 120; i++) {
    const v = dom[0] + ((dom[1] - dom[0]) * i) / 120;
    d += `${i ? 'L' : 'M'}${px(v).toFixed(1)},${py(f(v)).toFixed(1)}`;
  }
  // Tangent: a fixed screen length, slope in screen units.
  const dx = (SW - SP.l - SP.r) / (dom[1] - dom[0]);
  const dy = (-sl * k * (SH - SP.t - SP.b)) / yMax;
  const n = Math.hypot(dx, dy);
  const half = 70;
  const X0 = px(at);
  const Y0 = py(f(at));
  const tan = { x1: X0 - (dx / n) * half, y1: Y0 - (dy / n) * half, x2: X0 + (dx / n) * half, y2: Y0 + (dy / n) * half };
  const ticks = isW ? [0, 2, 4, 6, 8, 10] : [-4, 0, 4, 8, 12];
  return (
    <div style={{ opacity: Math.min(1, k * 2), transform: `translateX(${(1 - k) * -60}px)` }}>
      <div style={{ fontWeight: 700, color, fontSize: '1.05em', marginBottom: 2 }}>
        {isW ? tr(bi(`Only w moves (b = ${B0})`, `فقط w تکان می‌خورد (b = ${B0})`)) : tr(bi(`Only b moves (w = ${W0})`, `فقط b تکان می‌خورد (w = ${W0})`))}
      </div>
      <svg viewBox={`0 0 ${SW} ${SH}`} width="100%" style={{ display: 'block', direction: 'ltr' }} role="img" aria-label={`slice ${which}`}>
        <rect x={SP.l} y={SP.t} width={SW - SP.l - SP.r} height={SH - SP.t - SP.b} fill="none" stroke="var(--line)" />
        {ticks.map((v) => (
          <text key={v} x={px(v)} y={SH - SP.b + 18} textAnchor="middle" fontSize={13} fill="var(--muted)">
            {v}
          </text>
        ))}
        <text x={SW - SP.r} y={SH - 4} textAnchor="end" fontSize={17} fontWeight={700} fill={color}>
          {`${which} →`}
        </text>
        <text x={SP.l - 10} y={SP.t + 12} textAnchor="end" fontSize={17} fontWeight={700} fill={C.loss}>
          L
        </text>
        <path d={d} fill="none" stroke={C.loss} strokeWidth={3.5} />
        {k > 0.9 && (
          <g opacity={(k - 0.9) * 10}>
            <line {...tan} stroke={C.slope} strokeWidth={3.5} />
            <text x={X0 + 14} y={Y0 + 4} fontSize={17} fontWeight={800} fill={C.slope} className="halo">
              {/* the SVG is left-to-right, so the Persian label is written number first */}
              {tr(bi(`slope = ${fmt(sl, 2)}`, `${fmt(sl, 2)} = شیب`))}
            </text>
          </g>
        )}
        <circle cx={X0} cy={Y0} r={7} fill={C.weight} stroke="var(--surface)" strokeWidth={2} />
      </svg>
    </div>
  );
}

// ---------- the last step: a big network with many knobs ----------
function Network() {
  const layers = [6, 11, 11, 11, 4];
  const W = 900;
  const H = 400;
  const xs = layers.map((_, i) => 60 + (i * (W - 120)) / (layers.length - 1));
  const ys = (n: number) => Array.from({ length: n }, (_, j) => 30 + ((j + 0.5) * (H - 60)) / n);
  const edges: { x1: number; y1: number; x2: number; y2: number; i: number }[] = [];
  let i = 0;
  layers.slice(0, -1).forEach((n, l) => {
    ys(n).forEach((y1) => ys(layers[l + 1]).forEach((y2) => edges.push({ x1: xs[l], y1, x2: xs[l + 1], y2, i: i++ })));
  });
  // A fixed pseudo-random blink pattern.
  const rnd = (k: number) => ((Math.sin(k * 12.9898) * 43758.5453) % 1 + 1) % 1;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 900, display: 'block', direction: 'ltr' }} role="img" aria-label="a large network">
      {edges.map((e) => (
        <line key={e.i} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} stroke="var(--line-strong)" strokeWidth={1} opacity={0.6} />
      ))}
      {edges.map((e) => (
        <circle key={'k' + e.i} cx={e.x1 + (e.x2 - e.x1) * (0.2 + 0.6 * rnd(e.i + 11))} cy={e.y1 + (e.y2 - e.y1) * (0.2 + 0.6 * rnd(e.i + 11))} r={3.6} fill={C.weight} opacity={0.25}>
          <animate attributeName="opacity" values="0.15;1;0.15" dur={`${1.2 + 2 * rnd(e.i + 7)}s`} begin={`${3 * rnd(e.i + 3)}s`} repeatCount="indefinite" />
        </circle>
      ))}
      {layers.map((n, l) =>
        ys(n).map((y, j) => (
          <g key={`${l}-${j}`}>
            <circle cx={xs[l]} cy={y} r={11} fill="var(--surface)" stroke={l === 0 ? C.input : l === layers.length - 1 ? C.output : 'var(--ink-2)'} strokeWidth={2.5} />
            {l > 0 && (
              <circle cx={xs[l]} cy={y} r={4.5} fill={C.bias}>
                <animate attributeName="opacity" values="0.2;1;0.2" dur={`${1.4 + 2 * rnd(l * 31 + j)}s`} begin={`${3 * rnd(l * 17 + j)}s`} repeatCount="indefinite" />
              </circle>
            )}
          </g>
        )),
      )}
    </svg>
  );
}

// ---------- the scene ----------
function useAfter(active: boolean, ms: number) {
  const [on, setOn] = useState(false);
  useEffect(() => {
    if (!active) return setOn(false);
    const id = setTimeout(() => setOn(true), ms);
    return () => clearTimeout(id);
  }, [active, ms]);
  return on;
}

function KnobsAndMeter({ w, b, size = 86 }: { w: number; b: number; size?: number }) {
  return (
    <div dir="ltr" style={{ display: 'flex', gap: 18, alignItems: 'flex-end', justifyContent: 'center' }}>
      <Knob label="w" value={w} min={0} max={10} size={size} />
      <Knob label="b" value={b} min={-5} max={5} color={C.bias} size={size} />
      <Meter value={Math.round(loss2(w, b) * 1000) / 1000} max={METER_MAX} label="L" digits={2} />
    </div>
  );
}

function Side({ step, pos, tick }: { step: number; pos: Pt; tick: number }) {
  const { tr } = useT();
  const combine = useAfter(step === 5, 900);
  if (step === 1)
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 18, alignItems: 'center' }}>
        <KnobsAndMeter w={W0} b={B0} />
        <Note>{tr(bi('each spot = one w and one b · height = loss', 'هر نقطه = یک w و یک b · ارتفاع = خطا'))}</Note>
      </div>
    );
  if (step === 2)
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14, fontSize: '1.05em' }}>
        <Legend swatch={<span style={{ display: 'inline-block', width: 34, height: 3, background: C.loss }} />} text={tr(bi('equal height (equal loss)', 'ارتفاع یکسان (خطای یکسان)'))} />
        <Legend swatch={<span style={{ display: 'inline-block', width: 34, height: 5, background: C.loss }} />} text={tr(bi(`the line through us: L = ${L0}`, `خطی که از ما می‌گذرد: L = ${L0}`))} />
        <Legend swatch={<span style={{ display: 'inline-block', width: 34, borderTop: `3px dashed ${C.output}` }} />} text={tr(bi('valley floor: L = 0', 'کف دره: L = 0'))} />
        <Legend swatch={<span style={{ display: 'inline-block', width: 14, height: 14, borderRadius: 99, background: C.weight }} />} text={`w = ${W0}, b = ${B0}`} />
      </div>
    );
  if (step === 3 || step === 4)
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, width: '100%' }}>
        <Slice which="w" show />
        <div style={{ minHeight: 230 }}>{step === 4 && <Slice which="b" show />}</div>
      </div>
    );
  if (step === 5)
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center' }}>
        <div style={{ display: 'flex', gap: 14 }} dir="ltr">
          <ValueCard label={<Tex>{'\\partial L / \\partial \\cw{w}'}</Tex>} value={fmt(G0[0], 2)} color={C.slope} note={tr(bi('only w moves', 'فقط w'))} />
          <ValueCard label={<Tex>{'\\partial L / \\partial \\cb{b}'}</Tex>} value={fmt(G0[1], 2)} color={C.slope} note={tr(bi('only b moves', 'فقط b'))} />
        </div>
        <Appear show={combine} keep>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
            <div style={{ fontSize: '1.8em', color: 'var(--muted)', lineHeight: 1 }}>↓</div>
            <ValueCard big label={<>{tr(bi('gradient', 'گرادیان'))} <Tex>{'\\nabla L'}</Tex></>} value={<span dir="ltr">{`(${fmt(G0[0], 2)}, ${fmt(G0[1], 2)})`}</span>} color={C.slope} />
            <Note>{tr(bi(`the arrow is drawn ${1 / ARROW}× shorter`, `پیکان ${fa(1 / ARROW)} برابر کوتاه‌تر کشیده شده`))}</Note>
          </div>
        </Appear>
      </div>
    );
  if (step === 6) {
    const prev = DESC[Math.max(0, tick - 1)];
    const g = [dLdw(prev[0], prev[1]), dLdb(prev[0], prev[1])];
    const t3 = (v: number) => tex(Math.round(v * 1000) / 1000, 3);
    const p = (v: number) => (t3(v).startsWith('-') ? `(${t3(v)})` : t3(v));
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16, alignItems: 'center' }}>
        <KnobsAndMeter w={pos[0]} b={pos[1]} />
        <div style={{ fontSize: '1.25em' }}>
          <Tex display>{`\\cw{${t3(pos[0])}} = \\cw{${t3(prev[0])}} - \\clr{${ETA}} \\times \\cg{${p(g[0])}}`}</Tex>
          <Tex display>{`\\cb{${t3(pos[1])}} = \\cb{${t3(prev[1])}} - \\clr{${ETA}} \\times \\cg{${p(g[1])}}`}</Tex>
        </div>
        <Note>{tr(bi(`step ${tick} of ${DESC_N}`, `قدم ${fa(tick)} از ${fa(DESC_N)}`))}</Note>
      </div>
    );
  }
  return null;
}

function Legend({ swatch, text }: { swatch: ReactNode; text: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <span style={{ width: 36, display: 'inline-flex', justifyContent: 'center' }}>{swatch}</span>
      <span>{text}</span>
    </div>
  );
}

function Scene({ step }: { step: number }) {
  const { tr } = useT();
  const tick = useTicker(step === 6, DESC_N, 750);
  const showMap = useAfter(step === 2, 2100) || step >= 3;
  useEffect(() => {
    void loadPlotly(); // warm up while the first step is on screen
  }, []);

  if (step === 0) {
    return (
      <div className="stage">
        <div dir="ltr" style={{ display: 'flex', alignItems: 'center', gap: 28 }}>
          <Machine
            input={X}
            output={fmt(X * W0 + B0, 2)}
            rule="\hat y = \cw{w}\cdot x + \cb{b}"
            knobs={
              <>
                <Knob label="w" value={W0} min={0} max={10} size={76} />
                <Knob label="b" value={B0} min={-5} max={5} color={C.bias} size={76} />
              </>
            }
            inputLabel={tr(bi('notebooks', 'دفترچه'))}
            outputLabel={tr(bi('guess', 'حدس'))}
          />
          <Meter value={L0} max={METER_MAX} label="L" digits={2} />
        </div>
        <div style={{ fontSize: '1.5em' }}>
          <Tex display>{`\\cL{L}(\\cw{w}, \\cb{b}) = (2\\cw{w} + \\cb{b} - 10)^2 = (2\\cdot\\cw{${W0}} + \\cb{${B0}} - 10)^2 = \\cL{${L0}}`}</Tex>
        </div>
      </div>
    );
  }

  const pos: Pt = step === 6 ? DESC[tick] : step === 7 ? DESC[DESC_N] : START;
  const trail = step === 6 ? DESC.slice(0, tick + 1) : step === 7 ? DESC : [];
  const net = step === 7;
  return (
    <div style={{ flex: 1, position: 'relative', display: 'flex', minHeight: 0 }}>
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 28, opacity: net ? 0 : 1, transition: 'opacity 1s ease' }}>
        <div style={{ flex: '1 1 0', maxWidth: 640, minWidth: 0, position: 'relative' }}>
          {step <= 2 && (
            <div style={{ position: step === 2 && showMap ? 'absolute' : 'relative', inset: 0, opacity: step === 2 && showMap ? 0 : 1, transition: 'opacity 0.8s ease' }}>
              <Landscape mode={step === 2 ? 'top' : 'turn'} height={470} />
            </div>
          )}
          {step >= 2 && (
            <div style={{ opacity: showMap ? 1 : 0, transition: 'opacity 0.8s ease', position: showMap ? 'relative' : 'absolute', inset: 0 }}>
              <ContourMap pos={pos} trail={trail} slice={step === 3 ? 'w' : step === 4 ? 'b' : null} arrow={step === 5} />
            </div>
          )}
        </div>
        <div style={{ width: 420, flex: '0 0 auto' }}>{(step !== 2 || showMap) && <Side step={step} pos={pos} tick={tick} />}</div>
      </div>
      {net && (
        <div className="appear on" style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10 }}>
          <Network />
          <div style={{ fontSize: '1.4em' }}>
            <Tex display>{'\\cg{\\nabla L} = \\left(\\cg{\\frac{\\partial L}{\\partial w_1}},\\ \\cg{\\frac{\\partial L}{\\partial w_2}},\\ \\cg{\\frac{\\partial L}{\\partial w_3}},\\ \\dots,\\ \\cg{\\frac{\\partial L}{\\partial w_{1\\,000\\,000}}}\\right)'}</Tex>
          </div>
        </div>
      )}
    </div>
  );
}

export const gradientLesson: StoryLesson = {
  id: ID,
  Scene,
  steps: [
    {
      say: bi(
        'A real machine has several knobs. Let\'s bring back knob b. Now the loss depends on two things: w and b.',
        'ماشین واقعی چند پیچ دارد. پیچ b را برگردانیم. حالا خطا به دو چیز بستگی دارد: w و b.',
      ),
      more: bi(
        'This is the neuron from lesson 2: $\\hat y = w \\cdot x + b$, with w = 3 and b = 0. For 2 notebooks it still guesses 6, so the loss is still $(6 - 10)^2 = 16$. But now turning either knob changes the guess, so both knobs change the loss.',
        'این همان نورون درس ۲ است: $\\hat y = w \\cdot x + b$، با w = ۳ و b = ۰. برای ۲ دفترچه هنوز ۶ حدس می‌زند، پس خطا هنوز $(6 - 10)^2 = 16$ است. ولی حالا چرخاندن هر کدام از پیچ‌ها حدس را عوض می‌کند، پس هر دو پیچ روی خطا اثر دارند.',
      ),
    },
    {
      say: bi(
        'Instead of a curve we have a landscape: each point is a w and a b, and its height is the loss.',
        'به جای منحنی، یک زمین داریم: هر نقطه‌اش یک w و یک b است و ارتفاعش خطا.',
      ),
      more: bi(
        'With one knob we drew a curve: w to the right, loss up. With two knobs we need two directions on the ground, one for w and one for b, and the loss becomes the height above each spot. Our machine stands at w = 3, b = 0, at height 16.',
        'با یک پیچ یک منحنی کشیدیم: w به راست، خطا به بالا. با دو پیچ روی زمین دو جهت لازم داریم، یکی برای w و یکی برای b، و خطا می‌شود ارتفاعِ بالای هر نقطه. ماشین ما در w = ۳ و b = ۰ ایستاده، در ارتفاع ۱۶.',
      ),
    },
    {
      say: bi(
        'Look from above, like a hiking map: each line joins places of equal height.',
        'از بالا نگاهش کن، مثل نقشه کوهنوردی: هر خط، جاهای هم‌ارتفاع را به هم وصل می‌کند.',
      ),
      more: bi(
        'Seen from straight above, the height is gone, so we draw it with lines: walking along one line, the loss stays the same. The numbers on the lines are their heights. Here the valley is a long straight trough, so the lines are straight; the dashed line is its floor, where the loss is 0.',
        'از بالای بالا، ارتفاع دیده نمی‌شود، پس با خط نشانش می‌دهیم: اگر روی یک خط راه بروی، خطا عوض نمی‌شود. عددهای روی خط‌ها ارتفاعشان است. این‌جا دره یک گودالِ دراز و صاف است، برای همین خط‌ها راست‌اند؛ خط‌چین کف دره است، جایی که خطا صفر است.',
      ),
    },
    {
      say: bi(
        'Nudge only w, keep b fixed: the slope in the w direction is −16. The same derivative as in lesson 5.',
        'فقط w را تکان بده، b ثابت: شیب در جهت w = −۱۶. همان مشتق درس ۵.',
      ),
      recall: { lesson: 'derivative', note: bi('the slope of L(w) = (2w − 10)² at w = 3 is −16', 'شیب L(w) = (2w − 10)² در w = ۳ برابر −۱۶ است') },
      more: bi(
        'Keeping b at 0 means walking only along the purple line on the map. The heights along that line make the U curve from lesson 5, $(2w - 10)^2$, and its slope at w = 3 is −16. Nothing new: one knob at a time is just lesson 5 again.',
        'ثابت نگه داشتن b روی ۰ یعنی فقط روی خط بنفشِ نقشه راه برویم. ارتفاع‌ها در طول این خط همان منحنی U درس ۵ است، $(2w - 10)^2$، و شیبش در w = ۳ برابر −۱۶ است. چیز تازه‌ای نیست: هر بار یک پیچ، یعنی همان درس ۵.',
      ),
    },
    {
      say: bi('Now nudge only b: the slope in the b direction is −8.', 'حالا فقط b را تکان بده: شیب در جهت b = −۸.'),
      more: bi(
        `Now w stays at 3 and only b moves: we walk along the amber line. That slice is the curve $(b - 4)^2$, and at b = 0 its slope is −8. It is gentler than w's −16 because w is multiplied by 2 notebooks, while b is added only once.`,
        `حالا w روی ۳ می‌ماند و فقط b حرکت می‌کند: روی خط نارنجی راه می‌رویم. این برش منحنی $(b - 4)^2$ است و در b = ۰ شیبش −۸ است. از −۱۶ِ w ملایم‌تر است، چون w در ۲ دفترچه ضرب می‌شود ولی b فقط یک بار اضافه می‌شود.`,
      ),
    },
    {
      say: bi(
        'Each is a partial derivative (∂ means the others are held still). The list of both is the gradient: (−16, −8).',
        'به هر کدام «مشتق جزئی» می‌گوییم (∂ یعنی بقیه ثابت‌اند). فهرست هر دو با هم: گرادیان = (−۱۶، −۸).',
      ),
      more: bi(
        'The curly ∂ is just a reminder: "only this knob moved, the others stayed put". Put the two numbers side by side and you have the gradient, written $\\nabla L$. Drawn on the map as an arrow, it points the way the loss rises fastest: uphill.',
        'علامت ∂ فقط یک یادآوری است: «فقط همین پیچ تکان خورد، بقیه سر جایشان ماندند». دو عدد را کنار هم بگذار، می‌شود گرادیان که آن را $\\nabla L$ می‌نویسیم. اگر روی نقشه به شکل پیکان بکشیمش، به سمتی اشاره می‌کند که خطا در آن سریع‌تر از همه زیاد می‌شود: سربالایی.',
      ),
    },
    {
      say: bi(
        'Gradient descent is the same, for all knobs at once: each knob steps against its own slope.',
        'گرادیان کاهشی همان است، برای همه پیچ‌ها با هم: هر پیچ خلاف شیب خودش قدم برمی‌دارد.',
      ),
      more: bi(
        `The rule from lesson 6, once per knob, with η = ${ETA}: $w = 3 - ${ETA} \\times (-16) = ${fmt(DESC[1][0], 2)}$ and $b = 0 - ${ETA} \\times (-8) = ${fmt(DESC[1][1], 2)}$. Together that is one step against the gradient arrow, downhill. The loss falls from 16 to ${fmt(loss2(DESC[1][0], DESC[1][1]), 2)}, then keeps falling until the dot reaches the valley floor.`,
        `همان قاعده درس ۶، برای هر پیچ یک بار، با η = ۰٫۰۵: $w = 3 - ${ETA} \\times (-16) = ${fmt(DESC[1][0], 2)}$ و $b = 0 - ${ETA} \\times (-8) = ${fmt(DESC[1][1], 2)}$. این دو با هم یک قدم خلاف پیکان گرادیان است، رو به سرازیری. خطا از ۱۶ به ${fa(fmt(loss2(DESC[1][0], DESC[1][1]), 2))} می‌رسد و همین‌طور کم می‌شود تا نقطه به کف دره برسد.`,
      ),
    },
    {
      say: bi(
        'A real network has millions of knobs, so its gradient has millions of numbers. But how do we compute so many derivatives?',
        'یک شبکه واقعی میلیون‌ها پیچ دارد، پس گرادیانش میلیون‌ها عدد است. ولی این همه مشتق را چطور حساب کنیم؟',
      ),
      more: bi(
        'Every weight and every bias is a knob, and each one gets its own number in the gradient. Nudging the knobs one at a time and measuring would take forever. The next two lessons show a trick that gets all of them at once.',
        'هر وزن و هر بایاس یک پیچ است و هر کدام عدد خودش را در گرادیان دارد. اگر پیچ‌ها را یکی‌یکی تکان بدهیم و اندازه بگیریم، تا ابد طول می‌کشد. دو درس بعد ترفندی را نشان می‌دهند که همه را یک‌جا حساب می‌کند.',
      ),
    },
  ],
};
