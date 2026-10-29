// Lesson 6 · Walking downhill: gradient descent. Step against the slope; η sets the step size.
// Script: docs/storyboard.md, lesson 6. Running example L(w) = (2w − 10)², start w = 3.

import { useEffect, useRef, useState, type ReactNode } from 'react';
import { bi, useT, type Bi } from '../../i18n';
import { useUi } from '../../state/ui';
import { useLessonState } from '../../state/lessonState';
import { Tex } from '../../components/Tex';
import { Slider } from '../../components/controls';
import { fmt, tex } from '../../engine';
import type { StoryLesson } from '../types';
import { Appear, C, Knob, Machine, Meter, ValueCard } from '../kit';
import { useTween, useTicker } from '../useTween';
import { loss, path, slope, step as gdStep, stepsToReach, W0, W_BEST, X } from './math-descent';

const ID = 'descent';
const ETA = 0.05;
const ETA_SMALL = 0.005;
const ETA_BIG = 0.27;
const METER_MAX = 100; // the same range as lessons 4–5

const MAIN = path(W0, ETA, 60);
const AUTO_N = 10; // step 5: hops 1..AUTO_N
const SLOW_N = 30;
const SLOW = path(W0, ETA_SMALL, SLOW_N);
const FAST_N = 7; // the 7th jump leaves the plot
const FAST = path(W0, ETA_BIG, FAST_N);
const MAIN_TO_REACH = stepsToReach(W0, ETA, 0.01);
const GHOST = 7;
/** Persian digits for numbers inside Persian prose (not inside math). */
const fa = (v: number | string) => String(v).replace(/[0-9]/g, (d) => '۰۱۲۳۴۵۶۷۸۹'[Number(d)]).replace('.', '٫');
const TABLE_ROWS = 6;

// ---------- plot geometry ----------
const VW = 720;
const VH = 400;
const P = { l: 56, r: 20, t: 34, b: 46 };
const WD: [number, number] = [0, 10];
const sx = (w: number) => P.l + ((w - WD[0]) / (WD[1] - WD[0])) * (VW - P.l - P.r);
const inside = (w: number) => w >= WD[0] && w <= WD[1];
const hopHeight = (a: number, b: number) => Math.min(70, 10 + 0.35 * Math.abs(sx(b) - sx(a)));
const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

/** Moves x to `target` as a hop: returns the position, where it came from and the hop's progress. */
function useHop(target: number, ms: number) {
  const reduced = useUi((s) => s.reducedMotion);
  const [st, setSt] = useState({ x: target, from: target, t: 1 });
  const cur = useRef(target);
  useEffect(() => {
    if (reduced || ms <= 0 || cur.current === target) {
      cur.current = target;
      setSt({ x: target, from: target, t: 1 });
      return;
    }
    const from = cur.current;
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / ms);
      const x = from + (target - from) * ease(t);
      cur.current = x;
      setSt({ x, from, t });
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms, reduced]);
  return { ...st, to: target };
}

/** Screen geometry for a loss axis from 0 to yMax (it zooms out when the walker flies off). */
function makeGeo(yMax: number) {
  const sy = (L: number) => P.t + (1 - L / yMax) * (VH - P.t - P.b);
  /** Screen position of a hopping walker: a ballistic arc from the start point to the end point. */
  const hopXY = (h: { x: number; from: number; to: number; t: number }) => {
    const span = h.to - h.from;
    const k = span === 0 ? 1 : (h.x - h.from) / span;
    const y0 = sy(loss(h.from));
    const y1 = sy(loss(h.to));
    const lift = Math.sin(Math.PI * h.t);
    return { X: sx(h.x), Y: y0 + (y1 - y0) * k - hopHeight(h.from, h.to) * lift, lift };
  };
  /** A tangent segment of fixed screen length through w. */
  const tangent = (w: number, len = 150) => {
    const dx = (VW - P.l - P.r) / (WD[1] - WD[0]);
    const dy = (-slope(w) * (VH - P.t - P.b)) / yMax;
    const n = Math.hypot(dx, dy);
    const ux = (dx / n) * (len / 2);
    const uy = (dy / n) * (len / 2);
    const X0 = sx(w);
    const Y0 = sy(loss(w));
    return { x1: X0 - ux, y1: Y0 - uy, x2: X0 + ux, y2: Y0 + uy };
  };
  let curve = '';
  for (let i = 0; i <= 240; i++) {
    const w = WD[0] + ((WD[1] - WD[0]) * i) / 240;
    curve += `${i ? 'L' : 'M'}${sx(w).toFixed(1)},${sy(Math.min(loss(w), yMax * 1.5)).toFixed(1)}`;
  }
  const arc = (a: number, b: number) => {
    const x0 = sx(a);
    const y0 = sy(loss(a));
    const x1 = sx(b);
    const y1 = sy(loss(b));
    const h = hopHeight(a, b);
    return `M${x0},${y0} Q${(x0 + x1) / 2},${Math.min(y0, y1) - 2 * h} ${x1},${y1}`;
  };
  const step = yMax > 60 ? 25 : 10;
  const ticks = Array.from({ length: Math.floor(yMax / step) + 1 }, (_, i) => i * step);
  return { sy, hopXY, tangent, curve, arc, ticks };
}

/** Arrow pointing along the ground at height y from x, `dir` = ±1. */
function GroundArrow({ x, y, dir, color, label }: { x: number; y: number; dir: number; color: string; label?: string }) {
  const L = 74;
  const x2 = x + dir * L;
  return (
    <g>
      <line x1={x + dir * 16} y1={y} x2={x2} y2={y} stroke={color} strokeWidth={5} strokeLinecap="round" />
      <path d={`M${x2 + dir * 12},${y} L${x2},${y - 9} L${x2},${y + 9} Z`} fill={color} />
      {label && (
        <text x={x + dir * (L / 2 + 10)} y={y - 12} textAnchor="middle" fontSize={16} fontWeight={700} fill={color} className="halo">
          {label}
        </text>
      )}
    </g>
  );
}

interface CurveProps {
  walker: number;
  hopMs: number;
  fog: boolean;
  trail: number[];
  tangentAt: boolean;
  arrows: boolean;
  ghost: boolean;
  flyOff: boolean;
  showBest: boolean;
  /** Top of the loss axis. */
  yMax: number;
}

function Curve({ walker, hopMs, fog, trail, tangentAt, arrows, ghost, flyOff, showBest, yMax }: CurveProps) {
  const { tr, dir } = useT();
  const hop = useHop(walker, hopMs);
  const g = makeGeo(useTween(yMax, 1000));
  const { sy, tangent, arc } = g;
  const pos = g.hopXY(hop);
  const fogK = useTween(fog ? 1 : 0, 1100);
  const settled = hop.t >= 1;
  const sl = slope(walker);
  const t = tangent(walker);
  const gt = tangent(GHOST);
  const exitRight = walker > WD[1];
  return (
    <svg viewBox={`0 0 ${VW} ${VH}`} width="100%" style={{ maxWidth: 820, display: 'block', overflow: 'visible', direction: 'ltr' }} role="img" aria-label="loss curve">
      <defs>
        <clipPath id="ds-clip">
          <rect x={P.l} y={P.t} width={VW - P.l - P.r} height={VH - P.t - P.b} />
        </clipPath>
        <radialGradient id="ds-hole">
          <stop offset="0" stopColor="#000" />
          <stop offset="0.55" stopColor="#000" />
          <stop offset="1" stopColor="#fff" />
        </radialGradient>
        <mask id="ds-fog">
          <rect x={0} y={0} width={VW} height={VH} fill="#fff" />
          <circle cx={pos.X} cy={pos.Y} r={100} fill="url(#ds-hole)" />
          {ghost && <circle cx={sx(GHOST)} cy={sy(loss(GHOST))} r={100} fill="url(#ds-hole)" />}
        </mask>
        <filter id="ds-blur" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation={18} />
        </filter>
        <filter id="ds-shadow" x="-50%" y="-50%" width="200%" height="200%">
          <feDropShadow dx={0} dy={2} stdDeviation={2} floodOpacity={0.35} />
        </filter>
      </defs>
      <style>{'.halo{paint-order:stroke;stroke:var(--surface);stroke-width:4px;stroke-linejoin:round}'}</style>

      {/* axes and grid */}
      {g.ticks.map((v) => (
        <g key={'y' + v}>
          <line x1={P.l} x2={VW - P.r} y1={sy(v)} y2={sy(v)} stroke="var(--line)" />
          <text x={P.l - 8} y={sy(v) + 5} textAnchor="end" fontSize={14} fill="var(--muted)" className="num">
            {v}
          </text>
        </g>
      ))}
      {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((v) => (
        <g key={'x' + v}>
          <line x1={sx(v)} x2={sx(v)} y1={P.t} y2={VH - P.b} stroke="var(--line)" opacity={0.6} />
          <text x={sx(v)} y={VH - P.b + 20} textAnchor="middle" fontSize={14} fill="var(--muted)" className="num">
            {v}
          </text>
        </g>
      ))}
      <text x={VW - P.r} y={VH - 4} textAnchor="end" fontSize={18} fontWeight={700} fill={C.weight}>
        w →
      </text>
      <text x={P.l + 6} y={P.t - 12} textAnchor="start" fontSize={18} fontWeight={700} fill={C.loss}>
        L ↑
      </text>

      <g clipPath="url(#ds-clip)">
        <path d={g.curve} fill="none" stroke={C.loss} strokeWidth={4} strokeLinejoin="round" />
        {showBest && (
          <g>
            <line x1={sx(W_BEST)} x2={sx(W_BEST)} y1={sy(0)} y2={P.t + 30} stroke={C.output} strokeDasharray="5 5" strokeWidth={2} />
          </g>
        )}
        {/* the trail of hops */}
        {trail.slice(0, -1).map((w, i) => (
          <path key={'a' + i} d={arc(w, trail[i + 1])} fill="none" stroke={C.weight} strokeWidth={2} strokeDasharray="4 5" opacity={0.55} />
        ))}
        {trail.slice(0, -1).map((w, i) => (
          <circle key={'d' + i} cx={sx(w)} cy={sy(loss(w))} r={5} fill="var(--surface)" stroke={C.weight} strokeWidth={2.5} opacity={0.8} />
        ))}
        {/* the fog: covers everything except a patch around the walker */}
        {fogK > 0.01 && (
          <g mask="url(#ds-fog)" opacity={fogK}>
            <rect x={P.l} y={P.t} width={VW - P.l - P.r} height={VH - P.t - P.b} fill="var(--surface)" opacity={0.86} />
            <g filter="url(#ds-blur)" opacity={0.7}>
              <ellipse cx={200} cy={120} rx={160} ry={50} fill="var(--surface-2)">
                <animateTransform attributeName="transform" type="translate" values="0 0; 60 10; 0 0" dur="14s" repeatCount="indefinite" />
              </ellipse>
              <ellipse cx={520} cy={260} rx={190} ry={60} fill="var(--surface-2)">
                <animateTransform attributeName="transform" type="translate" values="0 0; -70 -8; 0 0" dur="17s" repeatCount="indefinite" />
              </ellipse>
            </g>
          </g>
        )}
        {/* the ghost walker at w = 7 */}
        {ghost && (
          <g opacity={0.6}>
            <line {...gt} stroke={C.slope} strokeWidth={3} />
            <text x={gt.x2 - 6} y={gt.y2 - 8} textAnchor="end" fontSize={15} fontWeight={700} fill={C.slope} className="halo">
              {`+${fmt(slope(GHOST), 2)}`}
            </text>
            <GroundArrow x={sx(GHOST)} y={sy(loss(GHOST)) - 26} dir={-1} color={C.output} />
            <circle cx={sx(GHOST)} cy={sy(loss(GHOST)) - 12} r={11} fill={C.weight} opacity={0.6} stroke="var(--surface)" strokeWidth={3} />
          </g>
        )}
        {/* tangent and direction at the walker */}
        {tangentAt && settled && inside(walker) && (
          <g>
            <line {...t} stroke={C.slope} strokeWidth={3.5} />
            <text x={pos.X} y={pos.Y - 60} textAnchor="middle" fontSize={16} fontWeight={700} fill={C.slope} className="halo">
              {`dL/dw = ${fmt(sl, 2)}`}
            </text>
          </g>
        )}
        {arrows && settled && Math.abs(sl) > 1e-9 && <GroundArrow x={pos.X} y={pos.Y - 26} dir={-Math.sign(sl)} color={C.output} />}
        {/* the walker */}
        <ellipse cx={sx(hop.x)} cy={sy(loss(hop.x)) + 3} rx={13 - 5 * pos.lift} ry={4} fill="var(--ink)" opacity={0.25 - 0.12 * pos.lift} />
        <circle cx={pos.X} cy={pos.Y - 12} r={11} fill={C.weight} stroke="var(--surface)" strokeWidth={3} filter="url(#ds-shadow)" />
      </g>
      {showBest && (
        <text x={sx(W_BEST)} y={P.t + 22} textAnchor="middle" direction={dir} fontSize={16} fontWeight={700} fill={C.output} className="halo">
          {tr(bi('bottom: w = 5', 'ته: w = 5'))}
        </text>
      )}
      {inside(walker) && (
        <text x={pos.X} y={VH - P.b + 38} textAnchor="middle" fontSize={15} fontWeight={700} fill={C.weight} opacity={settled ? 1 : 0.5}>
          {`w = ${fmt(hop.x, 2)}`}
        </text>
      )}
      {flyOff && !inside(walker) && settled && (
        <text x={exitRight ? VW - P.r - 8 : P.l + 8} y={P.t + 26} textAnchor={exitRight ? 'end' : 'start'} fontSize={20} fontWeight={800} fill={C.loss} className="halo">
          {exitRight ? tr(bi('flies off ↗', 'پرت شد ↗')) : tr(bi('↖ flies off', '↖ پرت شد'))}
        </text>
      )}
    </svg>
  );
}

// ---------- the update rule, written with coloured parts ----------
interface Part {
  tex: string;
  label?: Bi;
  color?: string;
}

function Parts({ parts, size = 1.7 }: { parts: Part[]; size?: number }) {
  const { tr } = useT();
  return (
    <div dir="ltr" style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'center', gap: 10, fontSize: `${size}em` }}>
      {parts.map((p, i) => (
        <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: p.label ? 64 : undefined }}>
          <Tex>{p.tex}</Tex>
          {p.label && (
            <span style={{ fontSize: '0.42em', fontWeight: 700, color: p.color, whiteSpace: 'nowrap', marginTop: 2 }} dir="auto">
              {tr(p.label)}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

const LBL = {
  next: bi('new w', 'w جدید'),
  cur: bi('w now', 'w فعلی'),
  eta: bi('step size', 'اندازه قدم'),
  slope: bi('slope', 'شیب'),
};
/** Rounded for display, never in scientific notation. */
const r3 = (v: number) => fmt(Math.round(v * 1000) / 1000, 3);
const t3 = (v: number) => tex(Math.round(v * 1000) / 1000, 3);
const paren = (v: number) => (t3(v).startsWith('-') ? `(${t3(v)})` : t3(v));

/** w_new = w − η × dL/dw, with numbers when `w` is given. */
function rule(eta?: number, w?: number): Part[] {
  const sym = eta === undefined || w === undefined;
  return [
    { tex: sym ? '\\cw{w_{\\text{new}}}' : `\\cw{${t3(gdStep(w, eta))}}`, label: LBL.next, color: C.weight },
    { tex: '=' },
    { tex: sym ? '\\cw{w}' : `\\cw{${t3(w)}}`, label: LBL.cur, color: C.weight },
    { tex: '-' },
    { tex: eta === undefined ? '\\clr{\\eta}' : `\\clr{${tex(eta, 3)}}`, label: LBL.eta, color: C.rate },
    { tex: '\\times' },
    { tex: sym ? '\\cg{\\dfrac{dL}{dw}}' : `\\cg{${paren(slope(w))}}`, label: LBL.slope, color: C.slope },
  ];
}

/** η as a small ruler: the length of one step. */
function Ruler() {
  return (
    <svg width={190} height={56} viewBox="0 0 190 56" aria-label="eta ruler">
      <text x={14} y={38} textAnchor="middle" fontSize={26} fontStyle="italic" fill={C.rate}>
        η
      </text>
      <rect x={34} y={14} width={150} height={30} rx={4} fill="var(--surface-2)" stroke={C.rate} strokeWidth={2} />
      {Array.from({ length: 11 }, (_, i) => (
        <line key={i} x1={41 + i * 13.6} x2={41 + i * 13.6} y1={14} y2={i % 5 === 0 ? 32 : 24} stroke={C.rate} strokeWidth={1.6} />
      ))}
    </svg>
  );
}

/** The last few hops (index, w, L); `first` is the index of the first row. */
function StepTable({ rows, first }: { rows: number[]; first: number }) {
  return (
    <table className="tbl" dir="ltr" style={{ fontSize: '0.95em', minWidth: 230 }}>
      <thead>
        <tr>
          <th className="num">#</th>
          <th className="num" style={{ color: C.weight }}>
            w
          </th>
          <th className="num" style={{ color: C.loss }}>
            L
          </th>
        </tr>
      </thead>
      <tbody>
        {rows.map((w, i) => (
          <tr key={i} style={{ fontWeight: i === rows.length - 1 ? 700 : 400 }}>
            <td className="num">{first + i}</td>
            <td className="num">{r3(w)}</td>
            <td className="num">{r3(loss(w))}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---------- the interactive last step ----------
interface Play {
  eta: number;
  hist: number[];
  running: boolean;
}
const PLAY0: Play = { eta: ETA, hist: [W0], running: false };
const lost = (w: number) => Math.abs(w - W_BEST) > 6;

function usePlay() {
  const [s, set] = useLessonState<Play>(ID, PLAY0);
  const doStep = () =>
    set((p) => {
      const w = p.hist[p.hist.length - 1];
      if (lost(w)) return { running: false };
      const n = gdStep(w, p.eta);
      const done = Math.abs(n - w) < 1e-4 || lost(n);
      return { hist: [...p.hist, n].slice(-60), running: p.running && !done };
    });
  useEffect(() => {
    if (!s.running) return;
    const id = setInterval(doStep, 550);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.running]);
  return { s, set, doStep, reset: () => set({ hist: [W0], running: false }) };
}

function Controls({ play }: { play: ReturnType<typeof usePlay> }) {
  const { tr } = useT();
  const { s, set, doStep, reset } = play;
  const w = s.hist[s.hist.length - 1];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, width: 270 }}>
      <Slider label={<Tex>{'\\clr{\\eta}'}</Tex>} value={s.eta} min={0.001} max={0.3} step={0.001} color="lr" digits={3} onChange={(eta) => set({ eta })} testId="descent-eta" />
      <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
        <button className="btn" onClick={doStep} disabled={lost(w)} data-testid="descent-step">
          {tr(bi('Step', 'قدم'))}
        </button>
        <button className="btn primary" onClick={() => set({ running: !s.running })} disabled={lost(w)} data-testid="descent-run">
          {s.running ? '❚❚' : '▶'} {tr(s.running ? bi('Pause', 'مکث') : bi('Run', 'اجرا'))}
        </button>
        <button className="btn" onClick={reset} data-testid="descent-reset">
          ↺ {tr(bi('Reset', 'از نو'))}
        </button>
      </div>
      <div className="scene-note">{`${tr(bi('steps', 'قدم'))}: ${s.hist.length - 1}`}</div>
    </div>
  );
}

// ---------- the scene ----------
function Scene({ step }: { step: number }) {
  const { tr } = useT();
  const play = usePlay();
  const autoTick = useTicker(step === 4, AUTO_N - 1, 600);
  const slowTick = useTicker(step === 6, SLOW_N, 380);
  const fastTick = useTicker(step === 7, FAST_N, 850);

  // Where the walker is, and the hops behind it, for this step.
  let trail: number[] = [W0];
  let hopMs = 650;
  let eta = ETA;
  if (step === 3) trail = MAIN.slice(0, 2);
  else if (step === 4) trail = MAIN.slice(0, autoTick + 2);
  else if (step === 5) trail = MAIN.slice(0, AUTO_N + 1).concat(MAIN[MAIN.length - 1]);
  else if (step === 6) {
    trail = SLOW.slice(0, slowTick + 1);
    hopMs = 300;
    eta = ETA_SMALL;
  } else if (step === 7) {
    trail = FAST.slice(0, fastTick + 1);
    hopMs = 700;
    eta = ETA_BIG;
  } else if (step === 8) {
    trail = play.s.hist;
    hopMs = 450;
    eta = play.s.eta;
  }
  const w = trail[trail.length - 1];
  const prev = trail.length > 1 ? trail[trail.length - 2] : undefined;

  let top: ReactNode = null;
  if (step <= 1)
    top = (
      <div style={{ fontSize: '1.6em' }}>
        <Tex>{'\\cL{L}(\\cw{w}) = (2\\cw{w} - 10)^2'}</Tex>
      </div>
    );
  else if (step === 2)
    top = (
      <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
        <Parts parts={rule()} />
        <Ruler />
      </div>
    );
  else if (step === 3) top = <Parts parts={rule(ETA, W0)} />;
  else if (step === 5)
    top = (
      <div style={{ transform: 'scale(0.56)', margin: '-56px 0' }}>
        <Machine input={X} output={fmt(X * w, 2)} rule="\hat y = \cw{w}\cdot x" knobs={<Knob label="w" value={w} min={0} max={10} size={70} />} pulse={step} inputLabel={tr(bi('notebooks', 'دفترچه'))} outputLabel={tr(bi('price', 'قیمت'))} />
      </div>
    );
  else top = <Parts parts={rule(eta, prev ?? w)} />;

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 6, minHeight: 0 }}>
      <div key={step === 5 ? 'm' : 'f'} className="appear on" style={{ minHeight: 92, display: 'flex', alignItems: 'center' }}>
        {top}
      </div>
      <div style={{ display: 'flex', gap: 24, alignItems: 'center', justifyContent: 'center', width: '100%' }}>
        <div style={{ flex: '1 1 0', maxWidth: 780, minWidth: 0 }}>
          <Curve
            walker={w}
            hopMs={hopMs}
            fog={step <= 4}
            trail={trail}
            tangentAt={step >= 1 && step !== 3 && step !== 5}
            arrows={step === 1 || step === 2}
            ghost={step === 1}
            flyOff={step >= 7}
            showBest={step >= 5}
            yMax={step >= 7 ? 110 : 40}
          />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, width: 280 }}>
          <div style={{ display: 'flex', gap: 14, alignItems: 'flex-end' }} dir="ltr">
            <Knob label="w" value={w} min={0} max={10} />
            <Meter value={Math.round(loss(w) * 1000) / 1000} max={METER_MAX} label="L" digits={2} />
          </div>
          <Appear show={step === 4}>
            <StepTable rows={trail.slice(-TABLE_ROWS)} first={Math.max(0, trail.length - TABLE_ROWS)} />
          </Appear>
          {step === 6 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <ValueCard label={`η = ${ETA}`} value={MAIN_TO_REACH} color={C.output} note={tr(bi('steps to reach w ≈ 5', 'قدم تا w ≈ 5'))} />
              <ValueCard label={`η = ${ETA_SMALL}`} value={slowTick} color={C.rate} note={`${tr(bi('steps so far', 'قدم تا این‌جا'))} · w = ${fmt(w, 2)}`} />
            </div>
          )}
          {step === 7 && <ValueCard label={`η = ${ETA_BIG}`} value={fastTick} color={C.loss} note={`${tr(bi('steps', 'قدم'))} · w = ${fmt(w, 2)}`} />}
          {step === 8 && <Controls play={play} />}
        </div>
      </div>
    </div>
  );
}

export const descentLesson: StoryLesson = {
  id: ID,
  Scene,
  steps: [
    {
      say: bi(
        'Imagine standing on a hill in thick fog. You can only feel the slope under your feet. How do you get down?',
        'فکر کن روی یک تپه‌ای و مه همه‌جا را گرفته. فقط شیب زیر پایت را حس می‌کنی. چطور به پایین برسی؟',
      ),
      more: bi(
        'The hill is the U curve from lesson 5: its height is the loss, and where you stand is the knob w = 3. The machine can\'t see the whole curve; it can only compute the slope right where it is. That is why the picture is foggy.',
        'این تپه همان منحنی U درس ۵ است: ارتفاعش خطاست و جایی که ایستاده‌ای پیچ w = ۳ است. ماشین کل منحنی را نمی‌بیند؛ فقط می‌تواند شیبِ همان جایی را که ایستاده حساب کند. برای همین تصویر مه‌آلود است.',
      ),
    },
    {
      say: bi('Step against the slope. Slope negative? Go forward. Positive? Go back.', 'خلاف شیب قدم بردار. شیب منفی است؟ برو جلو. مثبت است؟ برو عقب.'),
      more: bi(
        'At w = 3 the slope is −16: the ground goes down to the right, so we walk right (w gets bigger). At w = 7 the slope is +16: the ground goes down to the left, so we walk left. Either way, the green arrow points downhill.',
        'در w = ۳ شیب −۱۶ است: زمین به سمت راست پایین می‌رود، پس به راست می‌رویم (w بزرگ‌تر می‌شود). در w = ۷ شیب +۱۶ است: زمین به سمت چپ پایین می‌رود، پس به چپ می‌رویم. در هر دو حالت، پیکان سبز رو به سرازیری است.',
      ),
    },
    {
      say: bi('In maths: new w = w − η × derivative. η (eta) is the step size, called the learning rate.', 'به زبان ریاضی: w جدید = w − η × مشتق. η (اِتا) اندازه قدم است؛ به آن نرخ یادگیری می‌گوییم.'),
      more: bi(
        'The minus sign does the "against the slope" part for us: a negative slope makes w bigger, a positive slope makes it smaller. η is a small number we choose ourselves, like the length of our stride. A steeper slope also makes the step longer.',
        'علامت منفی کار «خلاف شیب» را خودش انجام می‌دهد: شیب منفی w را بزرگ‌تر می‌کند و شیب مثبت کوچک‌ترش. η یک عدد کوچک است که خودمان انتخابش می‌کنیم، مثل اندازه گام‌هایمان. شیب تندتر هم قدم را بلندتر می‌کند.',
      ),
    },
    {
      say: bi(
        'With η = 0.05: w = 3 − 0.05 × (−16) = 3.8. Subtracting a negative number adds. The loss drops from 16 to 5.76!',
        'با η = ۰٫۰۵: w = ۳ − ۰٫۰۵ × (−۱۶) = ۳٫۸. منهای یک عدد منفی یعنی اضافه کردن. خطا از ۱۶ رسید به ۵٫۷۶!',
      ),
      more: bi(
        `First $0.05 \\times (-16) = -0.8$. Then $3 - (-0.8) = 3 + 0.8 = 3.8$. Check it with the machine: 2 notebooks × 3.8 = 7.6, which is 2.4 short of 10, and $2.4^2 = 5.76$. One step, and the loss is already about a third.`,
        `اول $0.05 \\times (-16) = -0.8$. بعد $3 - (-0.8) = 3 + 0.8 = 3.8$. با ماشین امتحانش کن: ۲ دفترچه × ۳٫۸ = ۷٫۶، که ۲٫۴ تا از ۱۰ کمتر است، و $2.4^2 = 5.76$. فقط یک قدم، و خطا تقریباً یک‌سوم شده.`,
      ),
    },
    {
      say: bi(
        'Repeat: recompute the derivative each time and take another step. Near the bottom the slope is small, so the steps shrink.',
        'تکرار کن: هر بار مشتق را دوباره حساب کن و یک قدم بردار. نزدیک ته، شیب کم است، پس قدم‌ها کوچک می‌شوند.',
      ),
      more: bi(
        'The rule never changes: compute the slope where you stand, then step. At w = 3.8 the slope is −9.6, so the next step is only 0.48. Nobody tells the walker to slow down; the ground gets flatter, so the steps get shorter by themselves.',
        'قاعده هیچ‌وقت عوض نمی‌شود: شیب را همان جایی که ایستاده‌ای حساب کن، بعد قدم بردار. در w = ۳٫۸ شیب −۹٫۶ است، پس قدم بعدی فقط ۰٫۴۸ است. کسی به ماشین نمی‌گوید آهسته‌تر برو؛ زمین صاف‌تر می‌شود و قدم‌ها خودشان کوتاه‌تر می‌شوند.',
      ),
    },
    {
      say: bi(
        'We made it: w = 5. The machine says 10 for 2 notebooks, exactly right. It "learned" that a notebook costs 5.',
        'رسیدیم: w = ۵. ماشین برای ۲ دفترچه می‌گوید ۱۰، دقیقاً درست. ماشین «یاد گرفت» هر دفترچه ۵ سکه است.',
      ),
      more: bi(
        'At w = 5 the slope is 0, so the rule gives a step of 0 and the walker stays put. The fog lifts: the walker is at the very bottom of the curve. Nobody told the machine the answer 5; it found it just by following the slope.',
        'در w = ۵ شیب صفر است، پس قاعده یک قدم صفر می‌دهد و ماشین سر جایش می‌ماند. مه کنار می‌رود: ماشین درست در ته منحنی است. هیچ‌کس جواب ۵ را به ماشین نگفت؛ فقط با دنبال کردن شیب پیدایش کرد.',
      ),
    },
    {
      say: bi('If η is too small, the steps are tiny and it takes forever.', 'اگر η خیلی کوچک باشد، قدم‌ها ریزند و رسیدن خیلی طول می‌کشد.'),
      more: bi(
        `Same start, same rule, only η = 0.005. Each step is ten times shorter, so after ${SLOW_N} steps w is still only about ${fmt(SLOW[SLOW_N], 1)}. With η = 0.05 it took ${MAIN_TO_REACH} steps to get close to 5. It does get there, just very slowly.`,
        `همان شروع، همان قاعده، فقط η = ۰٫۰۰۵. هر قدم ده برابر کوتاه‌تر است، پس بعد از ${fa(SLOW_N)} قدم w هنوز فقط حدود ${fa(fmt(SLOW[SLOW_N], 1))} است. با η = ۰٫۰۵، ${fa(MAIN_TO_REACH)} قدم طول کشید تا به نزدیکی ۵ برسیم. بالاخره می‌رسد، ولی خیلی آهسته.`,
      ),
    },
    {
      say: bi(
        'If it\'s too big, you jump right over the bottom… and may land farther away each time.',
        'اگر خیلی بزرگ باشد، از ته دره می‌پری آن طرف... و ممکن است هر بار دورتر شوی.',
      ),
      more: bi(
        `With η = 0.27 the first step is $0.27 \\times 16 = 4.32$ long: from 3 to ${fmt(FAST[1], 2)}, far past the bottom at 5. There the ground is even steeper, so the next jump is even longer. The loss goes up instead of down, and the walker flies off the hill.`,
        `با η = ۰٫۲۷ قدم اول $0.27 \\times 16 = 4.32$ طول دارد: از ۳ به ${fa(fmt(FAST[1], 2))}، خیلی آن‌طرف‌ترِ ته دره در ۵. آن‌جا زمین حتی تندتر است، پس پرش بعدی بلندتر می‌شود. خطا به جای پایین آمدن بالا می‌رود و ماشین از تپه پرت می‌شود.`,
      ),
    },
    {
      say: bi('Try it yourself. Finding a good η is an important part of training a network.', 'خودت امتحان کن. پیدا کردن η خوب، بخش مهمی از آموزش شبکه است.'),
      more: bi(
        'Move the η slider, then press Step a few times or Run. Try 0.01, 0.1, 0.2 and 0.26. Which one gets to the bottom in the fewest steps? Which ones bounce? Reset starts again from w = 3.',
        'اسلایدر η را جابه‌جا کن، بعد چند بار «قدم» یا «اجرا» را بزن. ۰٫۰۱، ۰٫۱، ۰٫۲ و ۰٫۲۶ را امتحان کن. کدام با کمترین قدم به ته می‌رسد؟ کدام‌ها بالا و پایین می‌پرند؟ «از نو» دوباره از w = ۳ شروع می‌کند.',
      ),
    },
  ],
};
