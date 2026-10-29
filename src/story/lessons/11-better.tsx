// Lesson 11 · Better steps, a good start. A narrow valley makes plain steps zig-zag; momentum rolls
// through it; Adam is named; and the starting knobs must differ or neurons stay copies.
// Script: docs/storyboard.md, lesson 11. Numbers: math-better.ts (engine optimizerStep and trainStep).

import { useMemo, type ReactNode } from 'react';
import { bi, useT, type Bi } from '../../i18n';
import { Plot2D, type PlotLayer } from '../../components/Plot2D';
import { fmt, niceTicks, type OptimizerName } from '../../engine';
import type { StoryLesson } from '../types';
import { Appear, C, Meter, ValueCard } from '../kit';
import { useTicker, useTweenArray } from '../useTween';
import {
  BETA,
  HIDDEN,
  INIT_STEPS,
  OPTS,
  P_RANGE,
  Q_RANGE,
  RACE_STEPS,
  RANDOM,
  SAME,
  START,
  WAVE,
  distinctUnits,
  ellipseAt,
  momentumParts,
  output,
  pathOf,
  piece,
  runOf,
  showcaseStep,
  type StartId,
  type Unit,
} from './math-better';

const ID = 'better';
const S = { zigzag: 0, ball: 1, parts: 2, adam: 3, same: 4, random: 5 } as const;

const PATH_COLOR: Record<OptimizerName, string> = { sgd: 'var(--c-wneg)', momentum: C.input, adam: 'var(--c-a)' };
const PATH_NAME: Record<OptimizerName, Bi> = {
  sgd: bi('plain steps', 'قدم ساده'),
  momentum: bi('momentum', 'مومنتوم'),
  adam: bi('Adam', 'Adam'),
};
const LEVELS = [0.15, 0.6, 1.5, 3, 5.5, 9, 14];
const NEURON = ['var(--c-x)', 'var(--c-w)', 'var(--c-a)', 'var(--c-wneg)'];

// ---------------------------------------------------------------- the valley map

const W = 940;
const H = 420;
const PAD = { l: 46, r: 14, t: 12, b: 36 };
const PW = W - PAD.l - PAD.r;
const PH = H - PAD.t - PAD.b;
const FULL = [P_RANGE[0], P_RANGE[1], Q_RANGE[0], Q_RANGE[1]];

/** A view around the showcase step with equal scale on both axes, so arrow directions are true. */
function zoomView(k: number): number[] {
  const m = momentumParts(k);
  const prevStart: [number, number] = [m.at[0] - m.prev[0], m.at[1] - m.prev[1]];
  const tip: [number, number] = [m.at[0] + m.kept[0], m.at[1] + m.kept[1]];
  const end: [number, number] = [m.at[0] + m.next[0], m.at[1] + m.next[1]];
  const pts = [prevStart, m.at, tip, [tip[0] + m.push[0], tip[1] + m.push[1]], end];
  let [p0, p1, q0, q1] = [Math.min(...pts.map((p) => p[0])), Math.max(...pts.map((p) => p[0])), Math.min(...pts.map((p) => p[1])), Math.max(...pts.map((p) => p[1]))];
  const cp = (p0 + p1) / 2;
  const cq = (q0 + q1) / 2;
  let wp = (p1 - p0) * 1.7;
  let hq = (q1 - q0) * 1.7;
  if (wp / hq > PW / PH) hq = (wp * PH) / PW;
  else wp = (hq * PW) / PH;
  [p0, p1, q0, q1] = [cp - wp / 2, cp + wp / 2, cq - hq / 2, cq + hq / 2];
  return [p0, p1, q0, q1];
}

interface Arrow {
  from: [number, number];
  d: [number, number];
  color: string;
  width: number;
  dash?: string;
  opacity?: number;
  label?: string;
  /** Which side of the arrow the label sits on (+1 or −1). */
  side?: number;
}

function ValleyMap({ view, paths, arrows, ball }: { view: number[]; paths: { name: OptimizerName; upto: number; opacity: number }[]; arrows: Arrow[]; ball: boolean }) {
  const [p0, p1, q0, q1] = view;
  const sx = (p: number) => PAD.l + ((p - p0) / (p1 - p0)) * PW;
  const sy = (q: number) => PAD.t + (1 - (q - q0) / (q1 - q0)) * PH;
  const pt = niceTicks(p0, p1, 8);
  const qt = niceTicks(q0, q1, 6);
  const scale = PW / (p1 - p0);
  return (
    <div dir="ltr" style={{ width: '100%', maxWidth: W }}>
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W, display: 'block' }} role="img" aria-label="valley" data-testid="valley">
      <defs>
        <clipPath id="valley-clip">
          <rect x={PAD.l} y={PAD.t} width={PW} height={PH} />
        </clipPath>
        <marker id="valley-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="context-stroke" />
        </marker>
      </defs>
      <rect x={PAD.l} y={PAD.t} width={PW} height={PH} fill="var(--surface)" stroke="var(--line)" />
      {pt.map((t) => (
        <g key={'p' + t}>
          <line x1={sx(t)} x2={sx(t)} y1={PAD.t} y2={H - PAD.b} stroke="var(--line)" />
          <text x={sx(t)} y={H - PAD.b + 16} textAnchor="middle" fontSize={12} fill="var(--muted)" fontFamily="var(--font-mono)">
            {fmt(t, 2)}
          </text>
        </g>
      ))}
      {qt.map((t) => (
        <g key={'q' + t}>
          <line x1={PAD.l} x2={W - PAD.r} y1={sy(t)} y2={sy(t)} stroke="var(--line)" />
          <text x={PAD.l - 6} y={sy(t) + 4} textAnchor="end" fontSize={12} fill="var(--muted)" fontFamily="var(--font-mono)">
            {fmt(t, 2)}
          </text>
        </g>
      ))}
      <text x={PAD.l + PW / 2} y={H - 4} textAnchor="middle" fontSize={17} fontWeight={700} fill={C.weight}>
        p
      </text>
      <text x={14} y={PAD.t + PH / 2} textAnchor="middle" fontSize={17} fontWeight={700} fill={C.weight}>
        q
      </text>
      <g clipPath="url(#valley-clip)">
        {LEVELS.map((L) => {
          const [a, b] = ellipseAt(L);
          return <ellipse key={L} cx={sx(0)} cy={sy(0)} rx={a * scale} ry={(b * PH) / (q1 - q0)} fill="none" stroke="var(--line-strong)" strokeWidth={1.3} />;
        })}
        <text x={sx(0)} y={sy(0) + 5} textAnchor="middle" fontSize={16} fill={C.target}>
          ★
        </text>
        {paths.map(({ name, upto, opacity }) => {
          const pts = pathOf(name).points.slice(0, upto + 1);
          const last = pts[pts.length - 1];
          return (
            <g key={name} opacity={opacity}>
              <polyline points={pts.map((p) => `${sx(p[0])},${sy(p[1])}`).join(' ')} fill="none" stroke={PATH_COLOR[name]} strokeWidth={2.6} strokeLinejoin="round" />
              {pts.map((p, i) => (
                <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={2.6} fill={PATH_COLOR[name]} />
              ))}
              {ball && <circle cx={sx(last[0])} cy={sy(last[1])} r={9} fill={PATH_COLOR[name]} stroke="var(--surface)" strokeWidth={2.5} />}
            </g>
          );
        })}
        <circle cx={sx(START[0])} cy={sy(START[1])} r={7} fill="var(--surface)" stroke="var(--ink)" strokeWidth={2} />
        {arrows.map((a, i) => {
          const x1 = sx(a.from[0]);
          const y1 = sy(a.from[1]);
          const x2 = sx(a.from[0] + a.d[0]);
          const y2 = sy(a.from[1] + a.d[1]);
          const len = Math.hypot(x2 - x1, y2 - y1) || 1;
          const nx = (-(y2 - y1) / len) * (a.side ?? 1);
          const ny = ((x2 - x1) / len) * (a.side ?? 1);
          return (
            <g key={i} opacity={a.opacity ?? 1}>
              <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={a.color} strokeWidth={a.width} strokeDasharray={a.dash} markerEnd="url(#valley-arr)" strokeLinecap="round" />
              {a.label && (
                <text
                  x={(x1 + x2) / 2 + nx * 16}
                  y={(y1 + y2) / 2 + ny * 16 + 6}
                  textAnchor={Math.abs(nx) < 0.3 ? 'middle' : nx > 0 ? 'start' : 'end'}
                  fontSize={18}
                  fontWeight={700}
                  fill={a.color}
                  stroke="var(--surface)"
                  strokeWidth={5}
                  paintOrder="stroke"
                >
                  {a.label}
                </text>
              )}
            </g>
          );
        })}
      </g>
    </svg>
    </div>
  );
}

function Legend({ items }: { items: { name: OptimizerName; loss?: number }[] }) {
  const { tr } = useT();
  return (
    <div style={{ display: 'flex', gap: 26, justifyContent: 'center', flexWrap: 'wrap', fontSize: '1.05em' }}>
      {items.map(({ name, loss }) => (
        <span key={name} style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontWeight: 700, color: PATH_COLOR[name] }}>
          <i style={{ width: 26, height: 4, background: PATH_COLOR[name], borderRadius: 2, display: 'inline-block' }} />
          {tr(PATH_NAME[name])}
          {loss !== undefined && (
            <span className="num" style={{ color: C.loss, fontWeight: 600 }}>
              L = {fmt(loss, 3)}
            </span>
          )}
        </span>
      ))}
    </div>
  );
}

function Pill({ color, children }: { color: string; children: ReactNode }) {
  return <span style={{ border: `2px solid ${color}`, color, borderRadius: 10, padding: '4px 12px', fontWeight: 700, whiteSpace: 'nowrap' }}>{children}</span>;
}

function ValleyScene({ step }: { step: number }) {
  const { tr } = useT();
  const k = useMemo(() => showcaseStep(), []);
  const m = useMemo(() => momentumParts(k), [k]);
  const zoom = useMemo(() => zoomView(k), [k]);
  const tSgd = useTicker(step === S.zigzag, RACE_STEPS, 70);
  const tMom = useTicker(step === S.ball, RACE_STEPS, 70);
  const phase = useTicker(step === S.parts, 4, 1000);
  const tRace = useTicker(step === S.adam, RACE_STEPS, 85);
  const view = useTweenArray(step === S.parts ? zoom : FULL, 900);

  let paths: { name: OptimizerName; upto: number; opacity: number }[] = [];
  let arrows: Arrow[] = [];
  let legend: { name: OptimizerName; loss?: number }[] = [];
  if (step === S.zigzag) {
    paths = [{ name: 'sgd', upto: tSgd, opacity: 1 }];
    legend = [{ name: 'sgd', loss: pathOf('sgd').losses[tSgd] }];
  } else if (step === S.ball) {
    paths = [
      { name: 'sgd', upto: RACE_STEPS, opacity: 0.3 },
      { name: 'momentum', upto: tMom, opacity: 1 },
    ];
    legend = [{ name: 'sgd', loss: pathOf('sgd').losses[RACE_STEPS] }, { name: 'momentum', loss: pathOf('momentum').losses[tMom] }];
  } else if (step === S.parts) {
    paths = [{ name: 'momentum', upto: k, opacity: 0.45 }];
    const tip: [number, number] = [m.at[0] + m.kept[0], m.at[1] + m.kept[1]];
    const prevFrom: [number, number] = [m.at[0] - m.prev[0], m.at[1] - m.prev[1]];
    arrows = [{ from: prevFrom, d: m.prev, color: 'var(--muted)', width: 4, dash: '8 6', label: tr(bi('previous step', 'قدم قبلی')), side: 1 }];
    if (phase >= 1) arrows.push({ from: m.at, d: m.kept, color: C.input, width: 6, label: `${BETA} ×`, side: 1 });
    if (phase >= 2) arrows.push({ from: tip, d: m.push, color: C.slope, width: 6, label: tr(bi('new slope', 'شیب جدید')), side: 1 });
    if (phase >= 3) arrows.push({ from: m.at, d: m.next, color: C.output, width: 7, label: tr(bi('new step', 'قدم جدید')), side: -1 });
  } else {
    paths = OPTS.map((name) => ({ name, upto: tRace, opacity: 1 }));
    legend = OPTS.map((name) => ({ name, loss: pathOf(name).losses[tRace] }));
  }

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10, width: '100%' }}>
      <div style={{ minHeight: 34, display: 'flex', alignItems: 'center', gap: 22 }}>
        {step === S.parts ? (
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', fontSize: '1.1em', flexWrap: 'wrap', justifyContent: 'center' }}>
            <Pill color={C.output}>{tr(bi('new step', 'قدم جدید'))}</Pill>
            <span>=</span>
            <Pill color={C.input}>{tr(bi(`${BETA} × previous step`, `${BETA} × قدم قبلی`))}</Pill>
            <span>+</span>
            <Pill color={C.slope}>{tr(bi('new slope', 'شیب جدید'))}</Pill>
          </div>
        ) : (
          <>
            <Legend items={legend} />
            {step === S.adam && (
              <span style={{ fontWeight: 700, fontSize: '1.1em' }}>
                {tr(bi('step', 'قدم'))} <span className="num">{tRace} / {RACE_STEPS}</span>
              </span>
            )}
          </>
        )}
      </div>
      <ValleyMap view={view} paths={paths} arrows={arrows} ball={step !== S.parts} />
      {step === S.parts && (
        <div className="num" dir="ltr" style={{ color: 'var(--muted)', fontSize: '0.95em', textAlign: 'center' }}>
          p: {fmt(m.kept[0], 3)} + {m.push[0] < 0 ? `(${fmt(m.push[0], 3)})` : fmt(m.push[0], 3)} = {fmt(m.next[0], 3)} &nbsp;·&nbsp; q: {fmt(m.kept[1], 3)} + {m.push[1] < 0 ? `(${fmt(m.push[1], 3)})` : fmt(m.push[1], 3)} = {fmt(m.next[1], 3)}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- the start

const FRAME_EVERY = 4;

/** The tiny network 1 → 4 → 1; edge thickness shows each weight's size. */
function TinyNet({ units }: { units: Unit[] }) {
  const ys = [50, 120, 190, 260];
  const width = (v: number) => 1 + Math.min(9, Math.abs(v) * 4.5);
  return (
    <div dir="ltr">
    <svg viewBox="0 0 300 310" width={260} role="img" aria-label="network">
      {units.map((n, j) => (
        <g key={j}>
          <line x1={40} y1={155} x2={150} y2={ys[j]} stroke={NEURON[j]} strokeWidth={width(n.w)} opacity={0.75} />
          <line x1={150} y1={ys[j]} x2={260} y2={155} stroke={NEURON[j]} strokeWidth={width(n.u)} opacity={0.75} />
        </g>
      ))}
      <circle cx={40} cy={155} r={20} fill="var(--surface)" stroke={C.input} strokeWidth={3} />
      <text x={40} y={161} textAnchor="middle" fontSize={17} fontWeight={700} fill={C.input}>
        x
      </text>
      {units.map((_, j) => (
        <circle key={j} cx={150} cy={ys[j]} r={20} fill="var(--surface)" stroke={NEURON[j]} strokeWidth={4} />
      ))}
      <circle cx={260} cy={155} r={20} fill="var(--surface)" stroke={C.output} strokeWidth={3} />
      <text x={260} y={161} textAnchor="middle" fontSize={17} fontWeight={700} fill={C.output}>
        ŷ
      </text>
    </svg>
    </div>
  );
}

function StartScene({ start }: { start: StartId }) {
  const { tr } = useT();
  const run = useMemo(() => runOf(start), [start]);
  const n = Math.floor(INIT_STEPS / FRAME_EVERY);
  const tick = useTicker(true, n, 70);
  const k = Math.min(INIT_STEPS, tick * FRAME_EVERY);
  const units = run.units[k];
  const outB = run.outB[k];
  const distinct = distinctUnits(units);
  const layers: PlotLayer[] = [
    ...units.map((u, j): PlotLayer => ({ kind: 'fn', f: (x) => piece(u, x), color: NEURON[j], width: 2.4, dash: '7 5' })),
    { kind: 'fn', f: (x) => output(units, outB, x), color: C.output, width: 4 },
    { kind: 'points', color: C.target, pts: WAVE.X.map((x, i) => ({ x: x[0], y: WAVE.Y[i][0], r: 6.5 })) },
  ];
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12, width: '100%' }}>
      <div style={{ fontWeight: 700, fontSize: '1.1em', textAlign: 'center' }}>
        {start === 'same'
          ? tr(bi(`every weight starts at ${SAME} (biases at 0)`, `همه وزن‌ها از ${SAME} شروع می‌کنند (بایاس‌ها از 0)`))
          : tr(bi(`every weight starts at a random number between −${RANDOM} and ${RANDOM} (biases at 0)`, `هر وزن از یک عدد تصادفی بین −${RANDOM} و ${RANDOM} شروع می‌کند (بایاس‌ها از 0)`))}
        <div style={{ fontWeight: 500, fontSize: '0.88em', color: 'var(--muted)', textAlign: 'center', minHeight: '1.3em' }}>
          {distinct === 1 ? tr(bi('the 4 dashed curves lie exactly on top of each other', '۴ منحنی خط‌چین دقیقاً روی هم افتاده‌اند')) : tr(bi('4 different dashed curves', '۴ منحنی خط‌چین متفاوت'))}
        </div>
      </div>
      <div style={{ display: 'flex', gap: 22, alignItems: 'center', justifyContent: 'center', width: '100%' }}>
        <TinyNet units={units} />
        <div style={{ flex: '1 1 600px', maxWidth: 640, minWidth: 0 }}>
          <Plot2D
            xDomain={[-2.3, 2.3]}
            yDomain={[-1.8, 1.8]}
            layers={layers}
            height={290}
            xLabel="x"
            yLabel="ŷ"
            legend={[
              { label: tr(bi(`each neuron's piece (${HIDDEN})`, `سهم هر نورون (${HIDDEN})`)), color: NEURON[0], dash: true },
              { label: tr(bi('their sum: the output', 'جمعشان: خروجی')), color: C.output },
              { label: tr(bi('data', 'داده')), color: C.target },
            ]}
            testId="init-plot"
          />
        </div>
      </div>
      <div style={{ display: 'flex', gap: 30, alignItems: 'center', justifyContent: 'center' }}>
        <span style={{ fontWeight: 700, minWidth: 120, textAlign: 'center' }}>
          {tr(bi('step', 'قدم'))} <span className="num">{k} / {INIT_STEPS}</span>
        </span>
        <ValueCard label={tr(bi('different neurons', 'نورون‌های متفاوت'))} value={`${distinct} / ${HIDDEN}`} color={distinct === 1 ? C.loss : C.output} />
        <Meter value={Math.round(run.loss[k] * 1000) / 1000} max={0.7} label={tr(bi('loss', 'خطا'))} digits={3} />
      </div>
    </div>
  );
}

function Scene({ step }: { step: number }) {
  if (step <= S.adam) return <ValleyScene step={step} />;
  return (
    <Appear show>
      <StartScene key={step} start={step === S.same ? 'same' : 'random'} />
    </Appear>
  );
}

export const betterLesson: StoryLesson = {
  id: ID,
  Scene,
  steps: [
    {
      say: bi(
        'A real landscape is not always a round bowl. In a narrow valley, gradient descent zig-zags.',
        'زمین واقعی همیشه یک کاسه گرد نیست. در یک دره باریک، گرادیان کاهشی زیگزاگ می‌زند.',
      ),
      recall: { lesson: 'gradient', note: bi('the map of two knobs and the downhill direction', 'نقشه دو پیچ و جهت سرازیری') },
      more: bi(
        'This is a map of the loss for two knobs, p across and q up. Each ring is a line of equal loss, and the star is the bottom. Across q the walls are steep; along p the floor is almost flat. The downhill direction points mostly at the near wall, so each step jumps across the valley and back, and moves forward only a little.',
        'این نقشه خطا برای دو پیچ است: p به راست و q به بالا. هر حلقه یک خط با خطای برابر است و ستاره ته دره است. در جهت q دیواره‌ها تندند؛ در جهت p کف دره تقریباً صاف است. جهت سرازیری بیشتر به سمت دیواره نزدیک است، پس هر قدم از این طرف دره به آن طرف می‌پرد و برمی‌گردد و فقط کمی جلو می‌رود.',
      ),
    },
    {
      say: bi(
        "Instead of walking, let a ball roll: it builds up speed and smooths out the zig-zags. This is momentum.",
        'به جای قدم زدن، یک توپ رها کن: سرعت می‌گیرد و زیگزاگ‌ها را صاف می‌کند. به این «مومنتوم» می‌گوییم.',
      ),
      more: bi(
        'The blue path is a rolling ball. It still sways from side to side a little, like a real ball in a gutter, but smoothly, and the sways die out. Along the flat floor it keeps gaining speed, so in the same 60 steps it gets much closer to the star than the plain steps.',
        'مسیر آبی یک توپ غلتان است. هنوز کمی به چپ و راست تاب می‌خورد، مثل یک توپ واقعی در یک ناودان، ولی نرم، و تاب‌ها کم‌کم از بین می‌روند. روی کف صاف دره مدام سرعت می‌گیرد، پس در همان ۶۰ قدم خیلی نزدیک‌تر از قدم‌های ساده به ستاره می‌رسد.',
      ),
    },
    {
      say: bi(
        "Each step = 0.9 × the previous step + the new slope. It doesn't forget where it was going.",
        'هر قدم = ۰٫۹ قدم قبلی + شیب جدید. حرکت قبلی را فراموش نمی‌کند.',
      ),
      more: bi(
        'We zoomed in on one step of the ball. The grey dashed arrow is its previous step. The blue arrow keeps 0.9 of it. The pink arrow is the push from the new slope: a downhill step, as in lesson 6. Put them end to end and you get the green arrow, the new step. The pushes that flip back and forth cancel out, and the push along the valley keeps adding up.',
        'روی یک قدم توپ زوم کرده‌ایم. پیکان خاکستری خط‌چین قدم قبلی است. پیکان آبی ۰٫۹ آن را نگه می‌دارد. پیکان صورتی هُل شیب جدید است: یک قدم سرازیری، مثل درس ۶. این دو را پشت هم بگذار، پیکان سبز به دست می‌آید: قدم جدید. هُل‌هایی که هر بار جهتشان برعکس می‌شود همدیگر را خنثی می‌کنند و هُلی که در طول دره است مدام جمع می‌شود.',
      ),
    },
    {
      say: bi(
        'Adam does one more thing: it gives each knob its own step size. Most networks today train with Adam.',
        'روش Adam کار دیگری هم می‌کند: به هر پیچ اندازه قدم خودش را می‌دهد. امروز بیشتر شبکه‌ها با Adam آموزش می‌بینند.',
      ),
      more: bi(
        'Adam also remembers its previous steps, like the ball. In addition, a knob whose slope is always big gets smaller steps, and a knob whose slope is small gets bigger ones. So it moves quickly along the flat floor without crashing into the steep walls. All three start together; watch the step counter and each path\'s loss L.',
        'Adam هم مثل توپ قدم‌های قبلی‌اش را به یاد دارد. علاوه بر آن، پیچی که شیبش همیشه بزرگ است قدم‌های کوچک‌تر می‌گیرد و پیچی که شیبش کوچک است قدم‌های بزرگ‌تر. پس روی کف صاف سریع جلو می‌رود بدون اینکه به دیواره‌های تند بکوبد. هر سه با هم شروع می‌کنند؛ شمارنده قدم‌ها و خطای L هر مسیر را ببین.',
      ),
    },
    {
      say: bi(
        'Another question: where do the knobs start? If they all start the same, all neurons learn the same thing and stay copies forever.',
        'یک سؤال دیگر: پیچ‌ها از کجا شروع کنند؟ اگر همه یکسان شروع کنند، همه نورون‌ها یکسان یاد می‌گیرند و تا آخر کپی هم می‌مانند.',
      ),
      more: bi(
        'A tiny network with 4 hidden neurons learns the gold dots. Every weight starts at 0.5 (and every bias at 0). Each dashed curve is one neuron\'s piece, and the green curve is their sum. Same numbers give the same output, so backpropagation gives each the same slopes and the same update. The 4 curves stay on top of each other: 4 neurons doing the work of 1, and the wave is never fitted.',
        'یک شبکه کوچک با ۴ نورون پنهان نقطه‌های طلایی را یاد می‌گیرد. همه وزن‌ها از ۰٫۵ شروع می‌کنند (و همه بایاس‌ها از ۰). هر منحنی خط‌چین سهم یک نورون است و منحنی سبز جمع آن‌هاست. عددهای یکسان خروجی یکسان می‌دهند، پس پس‌انتشار به همه شیب یکسان و تغییر یکسان می‌دهد. ۴ منحنی روی هم می‌مانند: ۴ نورون کار ۱ نورون را می‌کنند و موج هیچ‌وقت درست یاد گرفته نمی‌شود.',
      ),
    },
    {
      say: bi(
        'So we start with small random numbers, and each neuron finds its own role.',
        'پس با عددهای کوچک و تصادفی شروع می‌کنیم تا هر نورون راه خودش را برود.',
      ),
      more: bi(
        'Same network, same data, same training. The only change: each weight starts at a different small random number. Now the neurons get different slopes, so they move apart, and each one bends at a different place. Together their pieces add up to the wave, and the loss drops close to 0.',
        'همان شبکه، همان داده، همان آموزش. فقط یک فرق: هر وزن از یک عدد تصادفی کوچکِ متفاوت شروع می‌کند. حالا نورون‌ها شیب‌های متفاوت می‌گیرند، پس از هم جدا می‌شوند و هر کدام در جای دیگری خم می‌شود. سهم‌هایشان با هم جمع می‌شوند و موج را می‌سازند و خطا تقریباً به ۰ می‌رسد.',
      ),
    },
  ],
};
