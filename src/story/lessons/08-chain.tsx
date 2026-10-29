// Lesson 8 · The chain rule. Through connected steps, rates multiply: gears first, then the path
// from w₁ to the loss in the two-neuron chain (preset "chain-two"), drawn as linked gears.
// Script: docs/storyboard.md, lesson 8.
//
// Also exports the chain-network drawing (ChainNet) and a clock hook, reused by lesson 9.

import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { bi, useT } from '../../i18n';
import { useLab } from '../../state/lab';
import { useUi } from '../../state/ui';
import { fmt, fmtSigned, tex } from '../../engine';
import type { StoryLesson } from '../types';
import { Appear, C, Formula, Stage, ValueCard } from '../kit';
import { chainOf, nudgeChain, type Chain } from './math-chain';

const ID = 'chain';

// ---------------------------------------------------------------------------------------------
// Shared pieces (also used by lesson 9).

/** Milliseconds since `active` became true (restarts when `resetKey` changes). Frozen at 0 when off. */
export function useClock(active: boolean, resetKey?: unknown): number {
  const reduced = useUi((s) => s.reducedMotion);
  const [t, setT] = useState(0);
  useEffect(() => {
    setT(0);
    if (!active) return;
    if (reduced) {
      // Reduced motion: jump straight to the end of every timeline.
      setT(1e7);
      return;
    }
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      setT(now - start);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active, resetKey, reduced]);
  return t;
}

/** The chain values for the network currently in the lab store. */
export function useChain(): Chain | null {
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const loss = useLab((s) => s.loss);
  const lossOpts = useLab((s) => s.lossOpts);
  return useMemo(() => {
    const isChain = net.inputSize === 1 && net.layers.length === 2 && net.layers.every((l) => l.b.length === 1);
    return isChain ? chainOf(net, data, loss, lossOpts) : null;
  }, [net, data, loss, lossOpts]);
}

/** A gear outline: n teeth between radius rRoot and rTip, first tooth centred at `phase` (radians). */
function gearPath(cx: number, cy: number, rRoot: number, rTip: number, n: number, phase = 0) {
  const s = (2 * Math.PI) / n;
  const pts: [number, number][] = [];
  const p = (a: number, r: number) => pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  for (let k = 0; k < n; k++) {
    const a = phase + k * s;
    p(a - 0.25 * s, rRoot);
    p(a - 0.13 * s, rTip);
    p(a + 0.13 * s, rTip);
    p(a + 0.25 * s, rRoot);
    p(a + 0.5 * s, rRoot);
  }
  return `M${pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join('L')}Z`;
}

/** A number for display: values too small to see at this precision show as 0 (no scientific notation in the scene). */
export const show = (v: number, digits = 2) => {
  if (!Number.isFinite(v)) return fmt(v, digits);
  const s = (Math.abs(v) < 0.5 * 10 ** -digits ? 0 : v).toFixed(digits);
  return (s.includes('.') ? s.replace(/\.?0+$/, '') : s).replace('-', '−');
};

const tint = (color: string, k = 16) => `color-mix(in srgb, ${color} ${k}%, var(--surface))`;

/** Shared layout of the chain network: x → w₁ → neuron 1 → w₂ → output → loss, in one row. */
export const NET_Y = 160;
export const VIEW_W = 980;
export const VIEW_H = 330;
export const POS = {
  x: 70,
  w1: 215,
  a1: 375,
  w2: 535,
  out: 695,
  loss: 900,
} as const;
export type NodeKey = keyof typeof POS;
export const RAD: Record<NodeKey, number> = {
  x: 40,
  w1: 32,
  a1: 46,
  w2: 32,
  out: 46,
  loss: 46,
};
const ORDER: NodeKey[] = ['x', 'w1', 'a1', 'w2', 'out', 'loss'];
const NODE_COLOR: Record<NodeKey, string> = {
  x: C.input,
  w1: C.weight,
  a1: 'var(--ink-2)',
  w2: C.weight,
  out: C.output,
  loss: C.loss,
};

export interface Chip {
  top: string;
  sub?: string;
  color: string;
  opacity?: number;
}
export interface EdgeStyle {
  color?: string;
  width?: number;
  dash?: string;
  offset?: number;
  opacity?: number;
}

/** A number chip drawn below a node (blame values, updates). */
function ChipView({ x, chip }: { x: number; chip: Chip }) {
  const w = Math.max(84, 16 + 15 * Math.max(chip.top.length, (chip.sub?.length ?? 0) * 0.62));
  const h = chip.sub ? 62 : 44;
  const y = NET_Y + 96;
  return (
    <g style={{ opacity: chip.opacity ?? 1, transition: 'opacity 0.5s' }}>
      <g className="ch-pop">
        <rect x={x - w / 2} y={y} width={w} height={h} rx={12} fill={tint(chip.color, 12)} stroke={chip.color} strokeWidth={2.5} />
        <text x={x} y={y + 31} textAnchor="middle" fontSize={26} fontWeight={800} fill={chip.color}>
          {chip.top}
        </text>
        {chip.sub && (
          <text x={x} y={y + 53} textAnchor="middle" fontSize={16} fontWeight={600} fill="var(--muted)">
            {chip.sub}
          </text>
        )}
      </g>
    </g>
  );
}

/** A rate badge above the network ("×4"), with an optional small note under it. */
export function RateBadge({ x, text, note, show, color = C.slope, y = 62 }: { x: number; text: string; note?: string; show: boolean; color?: string; y?: number }) {
  const w = Math.max(76, 22 * text.length);
  return (
    <g style={{ opacity: show ? 1 : 0, transition: 'opacity 0.5s' }}>
      <g className={show ? 'ch-pop' : undefined}>
        <rect x={x - w / 2} y={y - 23} width={w} height={46} rx={23} fill={tint(color, 14)} stroke={color} strokeWidth={2.5} />
        <text x={x} y={y + 10} textAnchor="middle" fontSize={28} fontWeight={800} fill={color}>
          {text}
        </text>
        {note && (
          <text x={x} y={y + 46} textAnchor="middle" fontSize={16} fontWeight={600} fill="var(--muted)">
            {note}
          </text>
        )}
      </g>
    </g>
  );
}

/**
 * The two-neuron chain, big and plain. Values come from `c` (the lab store). Gears, glows, chips,
 * a travelling pulse and extra overlays are switched on by each scene.
 */
export function ChainNet({
  c,
  teeth,
  edge,
  fade,
  glow,
  chips,
  pulse,
  children,
  maxWidth = 1060,
  digits = 2,
}: {
  c: Chain;
  /** Nodes drawn as gears, with their rotation in degrees. */
  teeth?: Partial<Record<NodeKey, number>>;
  /** Style of edge i (0: x–w₁, 1: w₁–neuron 1, 2: neuron 1–w₂, 3: w₂–output, 4: output–loss). */
  edge?: (i: number) => EdgeStyle | undefined;
  /** Node opacity (1 = normal). */
  fade?: Partial<Record<NodeKey, number>>;
  glow?: Partial<Record<NodeKey, string>>;
  chips?: Partial<Record<NodeKey, Chip>>;
  pulse?: { x: number; color: string; label?: string } | null;
  children?: ReactNode;
  maxWidth?: number;
  digits?: number;
}) {
  const { tr } = useT();
  const value: Record<NodeKey, number> = {
    x: c.x,
    w1: c.w1,
    a1: c.a1,
    w2: c.w2,
    out: c.yhat,
    loss: c.L,
  };
  // The loss gets one more digit, so a small loss near the end of training does not read as 0 too early.
  const text = Object.fromEntries(ORDER.map((k) => [k, show(value[k], k === 'loss' ? digits + 1 : digits)])) as Record<NodeKey, string>;
  const label: Record<NodeKey, string> = {
    x: tr(bi('input x', 'ورودی x')),
    w1: 'w₁',
    a1: tr(bi('neuron 1', 'نورون ۱')),
    w2: 'w₂',
    out: tr(bi('output ŷ', 'خروجی ŷ')),
    loss: tr(bi('loss L', 'خطا L')),
  };
  return (
    <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} width="100%" style={{ maxWidth, overflow: 'visible' }} role="img" aria-label="chain network" direction="ltr">
      <style>
        {
          '@keyframes ch-pop-k{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}} .ch-pop{animation:ch-pop-k .55s ease both} :root[data-motion=reduced] .ch-pop{animation:none}'
        }
      </style>
      {/* edges */}
      {ORDER.slice(0, -1).map((k, i) => {
        const a = ORDER[i + 1];
        const st = edge?.(i) ?? {};
        return (
          <line
            key={k}
            x1={POS[k] + RAD[k]}
            y1={NET_Y}
            x2={POS[a] - RAD[a]}
            y2={NET_Y}
            stroke={st.color ?? 'var(--line-strong)'}
            strokeWidth={st.width ?? 4}
            strokeDasharray={st.dash}
            strokeDashoffset={st.offset}
            strokeLinecap="round"
            opacity={st.opacity ?? 1}
            style={{
              transition: 'stroke 0.4s, opacity 0.4s, stroke-width 0.4s',
            }}
          />
        );
      })}
      {/* pulse (under the nodes, so values stay readable) */}
      {pulse && (
        <g>
          <circle cx={pulse.x} cy={NET_Y} r={30} fill={pulse.color} opacity={0.18} />
          <circle cx={pulse.x} cy={NET_Y} r={16} fill={pulse.color} opacity={0.9} />
        </g>
      )}
      {/* nodes */}
      {ORDER.map((k) => {
        const x = POS[k];
        const r = RAD[k];
        const col = NODE_COLOR[k];
        const rot = teeth?.[k];
        const isW = k === 'w1' || k === 'w2';
        return (
          <g key={k} opacity={fade?.[k] ?? 1} style={{ transition: 'opacity 0.5s' }}>
            {glow?.[k] && (
              <g className="ch-pop">
                <circle cx={x} cy={NET_Y} r={r + 20} fill={glow[k]} opacity={0.22} />
              </g>
            )}
            {rot !== undefined && (
              <g className="ch-pop">
                <path
                  d={gearPath(x, NET_Y, r + 2, r + 14, isW ? 10 : 14)}
                  transform={`rotate(${rot} ${x} ${NET_Y})`}
                  fill={tint(col, 30)}
                  stroke={col}
                  strokeWidth={2.5}
                  strokeLinejoin="round"
                />
              </g>
            )}
            <circle cx={x} cy={NET_Y} r={r} fill={isW ? tint(col, 10) : 'var(--surface)'} stroke={col} strokeWidth={isW ? 3 : 4} />
            <text x={x} y={NET_Y + 10} textAnchor="middle" fontSize={(isW ? 26 : 30) - (text[k].length > 4 ? 5 : 0)} fontWeight={800} fill={col}>
              {text[k]}
            </text>
            <text x={x} y={NET_Y + r + 30} textAnchor="middle" fontSize={19} fontWeight={600} fill={isW ? col : 'var(--muted)'}>
              {label[k]}
            </text>
          </g>
        );
      })}
      {/* the right answer, next to the loss */}
      <g>
        <rect x={POS.loss - 46} y={40} width={92} height={42} rx={10} fill={tint(C.target, 14)} stroke={C.target} strokeWidth={2.5} />
        <text x={POS.loss} y={69} textAnchor="middle" fontSize={22} fontWeight={800} fill={C.target}>
          y = {fmt(c.y, 2)}
        </text>
        <line x1={POS.loss} y1={82} x2={POS.loss} y2={NET_Y - RAD.loss} stroke={C.target} strokeWidth={2} strokeDasharray="4 4" />
      </g>
      {Object.entries(chips ?? {}).map(([k, chip]) => chip && <ChipView key={`${k}-${chip.top}`} x={POS[k as NodeKey]} chip={chip} />)}
      {pulse?.label && (
        <text x={pulse.x} y={NET_Y - 66} textAnchor="middle" fontSize={20} fontWeight={800} fill={pulse.color}>
          {pulse.label}
        </text>
      )}
      {children}
    </svg>
  );
}

// ---------------------------------------------------------------------------------------------
// Steps 1–2: three gears.

const TEETH = [36, 12, 6] as const;
const PITCH = 4.3;
const GEAR_TURN_MS = 6000;
const GEAR_PAUSE_MS = 1800;

function Gear({ cx, cy, n, phase, angle, color }: { cx: number; cy: number; n: number; phase: number; angle: number; color: string }) {
  const r = PITCH * n;
  return (
    <g transform={`rotate(${angle} ${cx} ${cy})`}>
      <path d={gearPath(cx, cy, r - 7, r + 7, n, phase)} fill={tint(color, 22)} stroke={color} strokeWidth={3} strokeLinejoin="round" />
      <circle cx={cx} cy={cy} r={Math.max(6, r * 0.22)} fill="var(--surface)" stroke={color} strokeWidth={3} />
      {/* a marker on the rim, so the turns can be counted */}
      <line x1={cx} y1={cy} x2={cx} y2={cy - r * 0.68} stroke={color} strokeWidth={4} strokeLinecap="round" />
      <circle cx={cx} cy={cy - r * 0.68} r={Math.max(5, r * 0.1)} fill={color} />
    </g>
  );
}

const GEAR_COLORS = [C.weight, 'var(--ink-2)', C.loss];

function GearScene({ step }: { step: number }) {
  const { tr } = useT();
  const reduced = useUi((s) => s.reducedMotion);
  const t = useClock(true);
  const period = GEAR_TURN_MS + GEAR_PAUSE_MS;
  const cycle = Math.floor(t / period);
  const tau = reduced ? 1 : Math.min(1, (t - cycle * period) / GEAR_TURN_MS);
  // Ease each turn so the start and stop are visible; the ratios stay exact at every moment.
  const e = tau < 0.5 ? 2 * tau * tau : 1 - (-2 * tau + 2) ** 2 / 2;
  const turns1 = reduced ? 0 : cycle + e;
  const [n1, n2, n3] = TEETH;
  const rate12 = n1 / n2;
  const rate23 = n2 / n3;
  const [r1, r2, r3] = TEETH.map((n) => PITCH * n);
  const cy = 200;
  const c1 = 40 + r1 + 8;
  const c2 = c1 + r1 + r2;
  const c3 = c2 + r2 + r3;
  const counts = [e, e * rate12, e * rate12 * rate23];
  const shown = (v: number, full: number) => (tau >= 1 ? full : Math.floor(v + 1e-9));
  const names = [bi('gear 1', 'چرخ ۱'), bi('gear 2', 'چرخ ۲'), bi('gear 3', 'چرخ ۳')];
  return (
    <Stage>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 36,
          flexWrap: 'wrap',
          justifyContent: 'center',
        }}
      >
        <svg viewBox={`0 0 ${c3 + r3 + 60} 400`} width={640} style={{ maxWidth: '100%' }} role="img" aria-label="gears" direction="ltr">
          <style>{'@keyframes ch-pop-k{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}} .ch-pop{animation:ch-pop-k .55s ease both}'}</style>
          <Gear cx={c1} cy={cy} n={n1} phase={0} angle={turns1 * 360} color={GEAR_COLORS[0]} />
          <Gear cx={c2} cy={cy} n={n2} phase={Math.PI + Math.PI / n2} angle={-turns1 * rate12 * 360} color={GEAR_COLORS[1]} />
          <Gear cx={c3} cy={cy} n={n3} phase={Math.PI} angle={turns1 * rate12 * rate23 * 360} color={GEAR_COLORS[2]} />
          {step >= 1 && (
            <>
              <RateBadge x={c2} y={cy - r2 - 52} text={`×${fmt(rate12)}`} show />
              <RateBadge x={c3 + 14} y={cy + r3 + 52} text={`×${fmt(rate23)}`} show />
            </>
          )}
        </svg>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {counts.map((v, i) => (
            <ValueCard key={i} label={tr(names[i])} value={shown(v, [1, rate12, rate12 * rate23][i])} color={GEAR_COLORS[i]} note={tr(bi('turns', 'دور'))} big={i === 2} />
          ))}
        </div>
      </div>
      <Appear show={step >= 1} keep>
        <Formula size={2.2}>{`\\cg{${tex(rate12)}} \\times \\cg{${tex(rate23)}} = \\cL{${tex(rate12 * rate23)}}`}</Formula>
      </Appear>
    </Stage>
  );
}

// ---------------------------------------------------------------------------------------------
// Steps 3–8: the path from w₁ to the loss, drawn as linked gears.

const BASE_TURN_MS = 8000;

function NetScene({ step }: { step: number }) {
  const { tr } = useT();
  const c = useChain();
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const loss = useLab((s) => s.loss);
  const lossOpts = useLab((s) => s.lossOpts);
  const gearsOn = step >= 3;
  const t = useClock(step >= 4);
  const nudge = useMemo(() => (c ? nudgeChain(net, data, 'w1', 0.001, loss, lossOpts) : null), [c, net, data, loss, lossOpts]);
  if (!c) return null;

  // Which links have their rate shown: the last from step 5, the middle from 6, the first from 7.
  const lastOn = step >= 4;
  const midOn = step >= 5;
  const firstOn = step >= 6;
  // Gear speeds, relative to the leftmost gear that is already linked.
  const speeds: Partial<Record<NodeKey, number>> = {};
  if (firstOn)
    Object.assign(speeds, {
      w1: 1,
      a1: c.rFirst,
      out: c.rFirst * c.rMid,
      loss: c.rFirst * c.rMid * c.rLast,
    });
  else if (midOn) Object.assign(speeds, { a1: 1, out: c.rMid, loss: c.rMid * c.rLast });
  else if (lastOn) Object.assign(speeds, { out: 1, loss: c.rLast });
  const turn = t / BASE_TURN_MS;
  const teeth: Partial<Record<NodeKey, number>> | undefined = gearsOn
    ? {
        w1: (speeds.w1 ?? 0) * turn * 360,
        a1: (speeds.a1 ?? 0) * turn * 360,
        out: (speeds.out ?? 0) * turn * 360,
        loss: (speeds.loss ?? 0) * turn * 360,
      }
    : undefined;
  const linkOn = [false, firstOn, midOn, midOn, lastOn];
  const linkSpeed = [0, speeds.a1 ?? 0, speeds.out ?? 0, speeds.out ?? 0, speeds.loss ?? 0];
  const edge = (i: number): EdgeStyle | undefined => {
    if (!gearsOn) return undefined;
    if (i === 0) return { opacity: 0.35 };
    return {
      color: linkOn[i] ? C.slope : 'var(--ink-2)',
      width: 7,
      dash: '14 9',
      offset: linkOn[i] ? -(turn * 60 * Math.min(linkSpeed[i], 6)) : 0,
    };
  };
  const chain = ['\\cg{' + tex(c.rLast) + '}', '\\cg{' + tex(c.rMid) + '}', '\\cg{' + tex(c.rFirst) + '}'].join(' \\times ');
  const relu2 = c.relu2 === 1 ? 'ReLU ×1' : `ReLU ×${fmt(c.relu2)}`;
  return (
    <Stage>
      <ChainNet
        c={c}
        teeth={teeth}
        edge={edge}
        fade={gearsOn ? { x: 0.4 } : undefined}
        chips={
          step === 2
            ? {
                loss: {
                  top: `(${fmt(c.yhat)} − ${fmt(c.y)})² = ${fmt(c.L)}`,
                  color: C.loss,
                },
              }
            : undefined
        }
      >
        {gearsOn && (
          <>
            <RateBadge
              x={(POS.w1 + POS.a1) / 2}
              text={firstOn ? `×${fmt(c.rFirst)}` : '×?'}
              note={firstOn ? `x = ${fmt(c.x)}` : undefined}
              show
              color={firstOn ? C.slope : 'var(--line-strong)'}
            />
            <RateBadge
              x={POS.w2}
              text={midOn ? `×${fmt(c.rMid)}` : '×?'}
              note={midOn ? `w₂ = ${fmt(c.w2)} · ${relu2}` : undefined}
              show
              color={midOn ? C.slope : 'var(--line-strong)'}
            />
            <RateBadge
              x={(POS.out + POS.loss) / 2 - 8}
              text={lastOn ? `×${fmt(c.rLast)}` : '×?'}
              note={lastOn ? `2 × (${fmt(c.yhat)} − ${fmt(c.y)})` : undefined}
              show
              color={lastOn ? C.slope : 'var(--line-strong)'}
            />
          </>
        )}
      </ChainNet>
      <Appear show={step >= 7} keep>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 28,
            flexWrap: 'wrap',
            justifyContent: 'center',
          }}
        >
          <Formula size={1.9}>{`\\frac{dL}{dw_1} = ${chain} = \\cg{${tex(c.gw1)}}`}</Formula>
          {nudge && (
            <ValueCard
              label={tr(bi('real nudge: w₁ + 0.001', 'تلنگر واقعی: w₁ + 0.001'))}
              value={`${fmtSigned(nudge.change, 4)}`}
              color={C.loss}
              note={<span dir="ltr">{`L: ${fmt(nudge.before, 3)} → ${fmt(nudge.after, 4)}`}</span>}
            />
          )}
        </div>
      </Appear>
    </Stage>
  );
}

function Scene({ step }: { step: number }) {
  // The lesson always shows the untouched chain-two network.
  useEffect(() => {
    const lab = useLab.getState();
    if (lab.presetId !== 'chain-two' || lab.stepper || lab.history.length > 1) lab.loadPreset('chain-two');
    if (lab.running) lab.setRunning(false);
  }, [step]);
  return step <= 1 ? <GearScene step={step} /> : <NetScene step={step} />;
}

export const chainLesson: StoryLesson = {
  id: ID,
  presetId: 'chain-two',
  Scene,
  steps: [
    {
      say: bi(
        'Three gears: turn the first once and the second turns 3 times. Each turn of the second turns the third twice. One turn of the first → 3 × 2 = 6 turns of the third.',
        'سه چرخ‌دنده: اولی یک دور بزند، دومی ۳ دور می‌زند. هر دور دومی، سومی را ۲ دور می‌چرخاند. اولی یک دور ← سومی ۳ × ۲ = ۶ دور.',
      ),
      more: bi(
        'Watch the dot on each gear and count its turns. The big gear has 3 times as many teeth as the middle one, so the middle one spins 3 times as fast. The small one spins twice as fast again, so it turns 6 times while the big one turns once.',
        'نقطه روی هر چرخ‌دنده را دنبال کن و دورهایش را بشمار. چرخ بزرگ ۳ برابرِ چرخ وسطی دندانه دارد، پس چرخ وسطی ۳ برابر تندتر می‌چرخد. چرخ کوچک هم دوباره ۲ برابرِ وسطی تند می‌چرخد، پس وقتی چرخ بزرگ یک دور می‌زند، چرخ کوچک ۶ دور می‌زند.',
      ),
    },
    {
      say: bi('The chain rule: when things are connected in a row, their rates multiply.', 'قانون زنجیره: وقتی چیزها پشت سر هم وصل‌اند، نرخ‌ها در هم ضرب می‌شوند.'),
      more: bi(
        'A rate says "how many times as much moves here, for each bit that moves there". Each gear passes the motion on and multiplies it by its own rate, so we don\'t add the rates, we multiply them: $3 \\times 2 = 6$.',
        'نرخ یعنی «به ازای هر مقدار حرکت آن‌جا، این‌جا چند برابرش حرکت می‌کند». هر چرخ حرکت را به بعدی می‌دهد و آن را در نرخ خودش ضرب می‌کند. پس نرخ‌ها را جمع نمی‌کنیم، در هم ضرب می‌کنیم: $3 \\times 2 = 6$.',
      ),
    },
    {
      say: bi(
        'Back to the two-neuron network from lesson 3: 2 → 6 → 12. The right answer is 10, so the loss is (12 − 10)² = 4.',
        'برگردیم سراغ شبکه دو نورونی درس ۳: ۲ ← ۶ ← ۱۲. جواب درست ۱۰ است، پس خطا = (۱۲ − ۱۰)² = ۴.',
      ),
      more: bi(
        'The input 2 is multiplied by w₁ = 3 in the first neuron and gives 6. The second neuron multiplies 6 by w₂ = 2 and gives 12. The right answer is 10, so the loss is $(12 - 10)^2 = 4$. Both ReLUs see positive numbers, so they let them through unchanged.',
        'ورودی ۲ در نورون اول در w₁ = 3 ضرب می‌شود و ۶ می‌دهد. نورون دوم ۶ را در w₂ = 2 ضرب می‌کند و ۱۲ می‌دهد. جواب درست ۱۰ است، پس خطا $(12 - 10)^2 = 4$ می‌شود. هر دو ReLU عدد مثبت می‌بینند، پس آن را بی‌تغییر رد می‌کنند.',
      ),
      recall: {
        lesson: 'network',
        note: bi('the two-neuron chain: 2 → 6 → 12', 'زنجیره دو نورونی: ۲ ← ۶ ← ۱۲'),
      },
    },
    {
      say: bi(
        "If we nudge w₁ a little, how much does the loss change? w₁ isn't wired straight to the loss; it goes through a chain.",
        'اگر w₁ را کمی تکان بدهیم، خطا چقدر عوض می‌شود؟ w₁ مستقیم به خطا وصل نیست؛ از یک زنجیره رد می‌شود.',
      ),
      more: bi(
        'Turning w₁ first changes what neuron 1 puts out, that changes the output, and that changes the loss. So the path is like a row of gears: w₁ → neuron 1 → output → loss. Each link has its own rate, still a question mark for now. We will find them one by one, starting from the end.',
        'چرخاندن w₁ اول خروجی نورون ۱ را عوض می‌کند، آن خروجی شبکه را عوض می‌کند و آن هم خطا را. پس مسیر مثل یک ردیف چرخ‌دنده است: w₁ ← نورون ۱ ← خروجی ← خطا. هر حلقه نرخ خودش را دارد که فعلاً علامت سؤال است. آن‌ها را یکی‌یکی پیدا می‌کنیم، از آخر شروع می‌کنیم.',
      ),
    },
    {
      say: bi(
        'Last link: nudge the output and the loss changes 4 times as much. (2 × (12 − 10) = 4)',
        'حلقه آخر: اگر خروجی یک تلنگر بخورد، خطا ۴ برابرش تغییر می‌کند. (۲ × (۱۲ − ۱۰) = ۴)',
      ),
      more: bi(
        'The loss is (output − 10)². Its slope is 2 times the gap, and the gap is $12 - 10 = 2$, so the rate is $2 \\times 2 = 4$. Now the last two gears turn together: the loss gear spins 4 times as fast as the output gear.',
        'خطا برابر (خروجی − ۱۰)² است. شیبش ۲ برابرِ فاصله است، و فاصله $12 - 10 = 2$ است، پس نرخ $2 \\times 2 = 4$ می‌شود. حالا دو چرخ آخر با هم می‌چرخند: چرخ خطا ۴ برابرِ چرخ خروجی تند می‌چرخد.',
      ),
    },
    {
      say: bi(
        'Middle link: the second neuron multiplies by 2, so the output changes twice as much as the first neuron does.',
        'حلقه وسط: نورون دوم ضرب در ۲ می‌کند، پس خروجی ۲ برابرِ تلنگر نورون اول تغییر می‌کند.',
      ),
      more: bi(
        "The second neuron takes what neuron 1 puts out and multiplies it by w₂ = 2. Its ReLU sees 12, a positive number, so it passes changes through as they are (×1). So this link's rate is $2 \\times 1 = 2$.",
        'نورون دوم خروجی نورون ۱ را می‌گیرد و در w₂ = 2 ضرب می‌کند. ReLU آن عدد ۱۲ را می‌بیند که مثبت است، پس تغییرها را همان‌طور رد می‌کند (×۱). پس نرخ این حلقه $2 \\times 1 = 2$ است.',
      ),
    },
    {
      say: bi('First link: the input is 2, so every nudge to w₁ is doubled in the first neuron.', 'حلقه اول: ورودی ۲ است، پس هر تلنگر به w₁ در نورون اول ۲ برابر می‌شود.'),
      more: bi(
        'Neuron 1 computes w₁ × 2. If w₁ grows by a little bit, the result grows by 2 times that bit. Its ReLU sees 6, which is positive, so nothing is blocked. Now every gear on the path is linked.',
        'نورون ۱ حساب می‌کند w₁ × 2. اگر w₁ کمی بزرگ شود، حاصل ۲ برابرِ همان مقدار بزرگ می‌شود. ReLU آن عدد ۶ را می‌بیند که مثبت است، پس چیزی جلویش را نمی‌گیرد. حالا همه چرخ‌های مسیر به هم وصل‌اند.',
      ),
    },
    {
      say: bi(
        "Multiply: 4 × 2 × 2 = 16. A real nudge agrees. It's the same derivative as in lesson 5, passed along a chain.",
        'ضرب کن: ۴ × ۲ × ۲ = ۱۶. تلنگر واقعی هم همین را می‌گوید. این همان مشتق درس ۵ است، فقط از یک زنجیره رد شده.',
      ),
      more: bi(
        'Multiplying the three rates gives $16$: when w₁ moves by a tiny amount, the loss moves 16 times as much. Check it for real: turn w₁ from 3 to 3.001, run the network again, and the loss goes from 4 to about 4.016, which is $16 \\times 0.001$.',
        'حاصل‌ضرب سه نرخ می‌شود $16$: وقتی w₁ کمی تکان بخورد، خطا ۱۶ برابرِ آن تکان می‌خورد. واقعاً امتحانش کن: w₁ را از ۳ به ۳٫۰۰۱ ببر و شبکه را دوباره اجرا کن؛ خطا از ۴ می‌رسد به حدود ۴٫۰۱۶، یعنی $16 \\times 0.001$.',
      ),
      recall: {
        lesson: 'derivative',
        note: bi('the derivative: how fast the loss changes when a knob turns', 'مشتق: وقتی پیچ می‌چرخد، خطا با چه سرعتی عوض می‌شود'),
      },
    },
  ],
};
