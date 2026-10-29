// Lesson 12 · Memorising or learning? Noisy data, a big network that memorises it, data set aside to
// check, stopping at the right time, using the trained machine, and a look back at the whole road.
// Script: docs/storyboard.md, lesson 12. Numbers: math-generalize.ts (engine trainStep on 'overfit').

import { useMemo, type ReactNode } from 'react';
import { bi, useT } from '../../i18n';
import { Plot2D, type PlotLayer } from '../../components/Plot2D';
import { fmt } from '../../engine';
import type { StoryLesson } from '../types';
import { CURRICULUM } from '../curriculum';
import { Appear, C, Knob, Machine, Meter, ValueCard } from '../kit';
import { useTicker, useTween } from '../useTween';
import { NEW_INPUTS, TRAIN_STEPS, asFn, fitStory, presetSizes, type FitStory } from './math-generalize';

const SIZES = presetSizes();

const ID = 'generalize';
const S = { noise: 0, memorise: 1, overfit: 2, stop: 3, use: 4, road: 5 } as const;
const CHECK = 'var(--c-a)';
const TRUTH = 'var(--ink-2)';
const METER_MAX = 0.3;
const X_DOM: [number, number] = [-2.2, 2.2];
const Y_DOM: [number, number] = [-1.9, 1.9];
/** Losses shown with 3 decimals, so a tiny loss reads 0 instead of a power of ten. */
const round3 = (v: number) => Math.round(v * 1000) / 1000;

/** Fit plot: the true rule, the measured dots, the checking dots and the network's curve. */
function FitPlot({ s, f, truth, wobble, check, misses, height = 400, extra = [] }: { s: FitStory; f?: (x: number) => number; truth: boolean; wobble: boolean; check: boolean; misses: boolean; height?: number; extra?: PlotLayer[] }) {
  const layers: PlotLayer[] = [];
  if (truth) layers.push({ kind: 'fn', f: s.truth, color: TRUTH, width: 2, dash: '7 6', opacity: 0.8 });
  if (wobble) s.data.X.forEach((x, n) => layers.push({ kind: 'segment', from: [x[0], s.truth(x[0])], to: [x[0], s.data.Y[n][0]], color: C.target, width: 2, opacity: 0.6 }));
  if (f) layers.push({ kind: 'fn', f, color: C.output, width: 3.5, samples: 400 });
  if (check && misses && f) s.val.X.forEach((x, n) => layers.push({ kind: 'segment', from: [x[0], f(x[0])], to: [x[0], s.val.Y[n][0]], color: C.loss, width: 1.6, dash: '3 3', opacity: 0.8 }));
  layers.push({ kind: 'points', color: C.target, pts: s.data.X.map((x, n) => ({ x: x[0], y: s.data.Y[n][0], r: 7 })) });
  if (check) layers.push({ kind: 'points', color: CHECK, pts: s.val.X.map((x, n) => ({ x: x[0], y: s.val.Y[n][0], r: 6.5, hollow: true })) });
  layers.push(...extra);
  return <Plot2D xDomain={X_DOM} yDomain={Y_DOM} layers={layers} height={height} xLabel="x" yLabel="y" testId="fit-plot" />;
}

function Key({ items }: { items: { color: string; text: string; kind: 'dot' | 'ring' | 'line' | 'dash' }[] }) {
  return (
    <div style={{ display: 'flex', gap: 18, justifyContent: 'center', flexWrap: 'wrap', fontSize: '0.98em', fontWeight: 600 }}>
      {items.map((it) => (
        <span key={it.text} style={{ display: 'inline-flex', alignItems: 'center', gap: 7, color: it.color }}>
          {it.kind === 'dot' && <i style={{ width: 13, height: 13, borderRadius: 99, background: it.color, display: 'inline-block' }} />}
          {it.kind === 'ring' && <i style={{ width: 11, height: 11, borderRadius: 99, border: `2.5px solid ${it.color}`, display: 'inline-block' }} />}
          {(it.kind === 'line' || it.kind === 'dash') && <i style={{ width: 26, borderTop: `3px ${it.kind === 'dash' ? 'dashed' : 'solid'} ${it.color}`, display: 'inline-block' }} />}
          {it.text}
        </span>
      ))}
    </div>
  );
}

/** Steps 1–3: the fit plot with the two loss meters. */
function FitScene({ s, step }: { s: FitStory; step: number }) {
  const { tr } = useT();
  const frames = s.nets.length - 1;
  const tick = useTicker(step === S.memorise, frames, 60);
  const k = step < S.memorise ? 0 : step === S.memorise ? tick : frames;
  const net = s.nets[k];
  const f = useMemo(() => asFn(net), [net]);
  const it = Math.min(TRAIN_STEPS, k * s.snap);
  const keys: { color: string; text: string; kind: 'dot' | 'ring' | 'line' | 'dash' }[] = [
    { color: TRUTH, text: tr(bi('the true rule', 'قاعده واقعی')), kind: 'dash' },
    { color: C.target, text: tr(bi('measured dots (with wobble)', 'نقطه‌های اندازه‌گیری‌شده (با نویز)')), kind: 'dot' },
  ];
  if (step >= S.memorise) keys.push({ color: C.output, text: tr(bi('the big network', 'شبکه بزرگ')), kind: 'line' });
  if (step >= S.overfit) keys.push({ color: CHECK, text: tr(bi('new dots', 'نقطه‌های جدید')), kind: 'ring' });
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10, width: '100%' }}>
      <Key items={keys} />
      <div style={{ display: 'flex', gap: 26, alignItems: 'center', justifyContent: 'center', width: '100%' }}>
        <div style={{ flex: '1 1 720px', maxWidth: 760, minWidth: 0 }}>
          <FitPlot s={s} f={step >= S.memorise ? f : undefined} truth wobble={step === S.noise} check={step >= S.overfit} misses={step >= S.overfit} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18, alignItems: 'center', minWidth: 170 }}>
          <Appear show={step >= S.memorise} keep>
            <Meter value={round3(step >= S.memorise ? s.trainLoss[it] : s.trainLoss[0])} max={METER_MAX} label={tr(bi('training loss', 'خطای آموزش'))} digits={3} />
          </Appear>
          <Appear show={step >= S.overfit} keep>
            <Meter value={round3(s.valLoss[TRAIN_STEPS])} max={METER_MAX} color={CHECK} label={tr(bi('loss on new dots', 'خطا روی نقطه‌های جدید'))} digits={3} />
          </Appear>
          {step === S.memorise && (
            <div style={{ color: 'var(--muted)', fontWeight: 600 }}>
              {tr(bi('step', 'قدم'))} <span className="num">{it}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/** Step 4: both losses over training time, and where to stop. */
function StopScene({ s }: { s: FitStory }) {
  const { tr } = useT();
  const frames = s.nets.length - 1;
  const tick = useTicker(true, frames + 3, 55);
  const k = Math.min(frames, tick);
  const done = tick > frames;
  const upto = k * s.snap;
  const toStop = useTween(done ? 1 : 0, 900);
  const fNow = useMemo(() => asFn(s.nets[k]), [s, k]);
  const fStop = useMemo(() => asFn(s.stopNet), [s]);
  const f = (x: number) => (1 - toStop) * fNow(x) + toStop * fStop(x);
  const train = s.trainLoss.slice(0, upto + 1).map((v, i) => [i, v] as [number, number]);
  const val = s.valLoss.slice(0, upto + 1).map((v, i) => [i, v] as [number, number]);
  const top = Math.max(s.trainLoss[0], s.valLoss[0]) * 1.08;
  const layers: PlotLayer[] = [
    { kind: 'polyline', pts: train, color: C.loss, width: 3 },
    { kind: 'polyline', pts: val, color: CHECK, width: 3 },
  ];
  if (done)
    layers.push(
      { kind: 'vline', at: s.stop, color: C.output, dash: '6 4' },
      { kind: 'points', color: C.output, pts: [{ x: s.stop, y: s.valLoss[s.stop], r: 8 }] },
    );
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10, width: '100%' }}>
      <Key
        items={[
          { color: C.loss, text: tr(bi('loss on the training dots', 'خطا روی نقطه‌های آموزش')), kind: 'line' },
          { color: CHECK, text: tr(bi('loss on the dots set aside to check', 'خطا روی نقطه‌های کنار گذاشته')), kind: 'line' },
        ]}
      />
      <div style={{ display: 'flex', gap: 20, alignItems: 'flex-start', justifyContent: 'center', width: '100%' }}>
        <div style={{ flex: '1 1 560px', maxWidth: 600, minWidth: 0 }}>
          <Plot2D xDomain={[0, TRAIN_STEPS]} yDomain={[0, top]} layers={layers} height={320} xLabel={tr(bi('training step', 'قدم آموزش'))} yLabel="L" testId="loss-curves" />
          <div style={{ textAlign: 'center', minHeight: 30, fontWeight: 700, fontSize: '1.1em', color: C.output }}>
            {done && (
              <>
                ▲ {tr(bi('stop here: step', 'همین‌جا متوقف کن: قدم'))} <span className="num">{s.stop}</span>
              </>
            )}
          </div>
        </div>
        <div style={{ flex: '1 1 440px', maxWidth: 480, minWidth: 0 }}>
          <FitPlot s={s} f={f} truth={false} wobble={false} check misses={false} height={320} />
          <div style={{ textAlign: 'center', color: 'var(--muted)', fontWeight: 600 }}>
            {done ? tr(bi('the network at the stopping step', 'شبکه در قدم توقف')) : `${tr(bi('step', 'قدم'))} ${upto}`}
          </div>
        </div>
      </div>
    </div>
  );
}

/** Step 5: the trained machine, knobs locked, answering new inputs. */
function UseScene({ s }: { s: FitStory }) {
  const { tr } = useT();
  const f = useMemo(() => asFn(s.stopNet), [s]);
  const tick = useTicker(true, NEW_INPUTS.length, 1500);
  const shown = Math.min(NEW_INPUTS.length, tick + 1);
  const x = NEW_INPUTS[shown - 1];
  const W1 = s.stopNet.layers[0].W;
  const knobs = (
    <>
      {[0, 1, 2].map((j) => (
        <div key={j} style={{ fontSize: '0.6em', padding: '0 4px' }}>
          <Knob label={`w${j + 1}`} value={W1[j][0]} min={-3} max={3} size={62} />
        </div>
      ))}
      <span style={{ alignSelf: 'center', fontSize: '0.6em', color: 'var(--muted)' }}>+{s.knobs - 3}</span>
    </>
  );
  const answers: PlotLayer[] = [
    ...NEW_INPUTS.slice(0, shown).flatMap((v): PlotLayer[] => [
      { kind: 'segment', from: [v, Y_DOM[0]], to: [v, f(v)], color: C.input, dash: '5 4', width: 1.6 },
      { kind: 'points', color: C.output, pts: [{ x: v, y: f(v), r: 8 }] },
    ]),
  ];
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8, width: '100%' }}>
      <div style={{ transform: 'scale(0.82)', margin: '-14px 0' }}>
        <Machine input={<span style={{ fontSize: '0.8em' }}>{fmt(x, 2)}</span>} output={<span style={{ fontSize: '0.8em' }}>{fmt(f(x), 2)}</span>} rule="\hat{y} = f(x)" knobs={knobs} pulse={shown} locked inputLabel={tr(bi('new input', 'ورودی جدید'))} outputLabel={tr(bi('answer', 'جواب'))} />
      </div>
      <div style={{ display: 'flex', gap: 22, alignItems: 'center', justifyContent: 'center', width: '100%' }}>
        <div style={{ flex: '1 1 620px', maxWidth: 660, minWidth: 0 }}>
          <Plot2D xDomain={X_DOM} yDomain={Y_DOM} layers={[{ kind: 'fn', f, color: C.output, width: 3.5 }, ...answers]} height={250} xLabel="x" yLabel="ŷ" testId="use-plot" />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, opacity: 0.45 }}>
          <ValueCard label={tr(bi('loss', 'خطا'))} value="—" color={C.loss} note={tr(bi('not needed', 'لازم نیست'))} />
          <ValueCard label={tr(bi('derivative', 'مشتق'))} value="—" color={C.slope} note={tr(bi('not needed', 'لازم نیست'))} />
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- the road

const ICON: ReactNode[] = [
  // 1 machine: a box with an arrow in and out
  <g key="1">
    <rect x={18} y={16} width={28} height={24} rx={4} />
    <path d="M4 28h12M48 28h12" />
    <path d="M12 24l4 4-4 4M56 24l4 4-4 4" />
  </g>,
  // 2 neuron: a knob
  <g key="2">
    <circle cx={32} cy={30} r={16} />
    <path d="M32 30l9-9" />
    <path d="M18 46a20 20 0 0 1-4-6M46 46a20 20 0 0 0 4-6" />
  </g>,
  // 3 network: three layers of nodes
  <g key="3">
    <path d="M12 30L32 16M12 30L32 44M32 16L52 30M32 44L52 30M12 30h0" />
    <circle cx={12} cy={30} r={5} />
    <circle cx={32} cy={16} r={5} />
    <circle cx={32} cy={44} r={5} />
    <circle cx={52} cy={30} r={5} />
  </g>,
  // 4 loss: a meter
  <g key="4">
    <path d="M10 42a22 22 0 0 1 44 0" />
    <path d="M32 42l12-14" />
    <circle cx={32} cy={42} r={2.5} />
  </g>,
  // 5 derivative: a curve and its tangent
  <g key="5">
    <path d="M8 48C20 46 30 38 36 26S48 10 56 8" />
    <path d="M18 52L52 18" strokeDasharray="4 3" />
    <circle cx={34} cy={30} r={3} />
  </g>,
  // 6 downhill: a ball on a slope
  <g key="6">
    <path d="M6 14c10 0 16 30 26 30s16-20 26-20" />
    <circle cx={20} cy={24} r={5} />
    <path d="M24 30l4 6" />
  </g>,
  // 7 gradient: contour rings and an arrow
  <g key="7">
    <ellipse cx={32} cy={30} rx={22} ry={14} />
    <ellipse cx={32} cy={30} rx={10} ry={6} />
    <path d="M50 18L38 26M38 26l1-6M38 26l6 1" />
  </g>,
  // 8 chain: two links
  <g key="8">
    <rect x={6} y={22} width={28} height={16} rx={8} />
    <rect x={30} y={22} width={28} height={16} rx={8} />
  </g>,
  // 9 backprop: nodes with a backward arrow
  <g key="9">
    <circle cx={12} cy={22} r={5} />
    <circle cx={32} cy={22} r={5} />
    <circle cx={52} cy={22} r={5} />
    <path d="M17 22h10M37 22h10" />
    <path d="M50 42H14M14 42l6-5M14 42l6 5" />
  </g>,
  // 10 data: dots and a line
  <g key="10">
    <path d="M8 50L56 12" />
    <circle cx={14} cy={40} r={3} />
    <circle cx={24} cy={38} r={3} />
    <circle cx={32} cy={26} r={3} />
    <circle cx={44} cy={24} r={3} />
    <circle cx={50} cy={12} r={3} />
  </g>,
  // 11 better steps: a smooth path to a star
  <g key="11">
    <path d="M6 14c8 10 12 30 22 26s10-18 18-14 6 6 8 6" />
    <path d="M56 28l2 4 4 1-3 3 1 4-4-2-4 2 1-4-3-3 4-1z" />
  </g>,
  // 12 real use: a lock
  <g key="12">
    <rect x={16} y={28} width={32} height={22} rx={4} />
    <path d="M22 28v-6a10 10 0 0 1 20 0v6" />
    <circle cx={32} cy={39} r={3} />
  </g>,
];

function RoadScene() {
  const { tr } = useT();
  const tick = useTicker(true, CURRICULUM.length, 420);
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 18, width: '100%' }} data-testid="road">
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, minmax(0, 150px))', gap: '22px 18px', justifyContent: 'center', width: '100%' }}>
        {CURRICULUM.map((l, i) => {
          const on = i < tick;
          const now = i === tick - 1;
          return (
            <div
              key={l.id}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 6,
                padding: '12px 6px 10px',
                borderRadius: 14,
                border: `2px solid ${on ? 'var(--accent)' : 'var(--line)'}`,
                background: now ? 'var(--accent-soft)' : 'var(--surface)',
                opacity: on ? 1 : 0.22,
                transform: now ? 'scale(1.08)' : 'none',
                transition: 'opacity 0.4s, transform 0.4s, background 0.4s, border-color 0.4s',
              }}
            >
              <span className="num" style={{ fontSize: '0.8em', fontWeight: 700, color: 'var(--accent)' }}>
                {i + 1}
              </span>
              <svg viewBox="0 0 64 60" width={84} height={78} fill="none" stroke={on ? 'var(--accent)' : 'var(--muted)'} strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                {ICON[i]}
              </svg>
              <span style={{ fontWeight: 700, fontSize: '1.05em', textAlign: 'center' }}>{tr(l.short)}</span>
            </div>
          );
        })}
      </div>
      <Appear show={tick >= CURRICULUM.length}>
        <div style={{ fontSize: '1.25em', fontWeight: 700, textAlign: 'center' }}>{tr(bi('one machine, a few ideas', 'یک ماشین، چند ایده'))}</div>
      </Appear>
    </div>
  );
}

function Scene({ step }: { step: number }) {
  const s = useMemo(() => fitStory(), []);
  if (step <= S.overfit) return <FitScene s={s} step={step} />;
  if (step === S.stop) return <StopScene s={s} />;
  if (step === S.use) return <UseScene s={s} />;
  return <RoadScene />;
}

export const generalizeLesson: StoryLesson = {
  id: ID,
  presetId: 'overfit',
  Scene,
  steps: [
    {
      say: bi(
        'Real data is a little noisy, like a scale that reads slightly differently every time.',
        'داده‌های واقعی کمی نویز دارند، مثل ترازویی که هر بار کمی فرق نشان می‌دهد.',
      ),
      more: bi(
        'The dashed curve is the true rule behind the data. Nobody gives it to us; we only get the gold dots, which are measurements. Each one is a little above or below the curve, like weighing the same bag twice and getting two slightly different numbers. That wobble is called noise.',
        'منحنی خط‌چین قاعده واقعی پشت داده‌هاست. کسی آن را به ما نمی‌دهد؛ ما فقط نقطه‌های طلایی را داریم که اندازه‌گیری‌اند. هر کدام کمی بالاتر یا پایین‌تر از منحنی است، مثل اینکه یک کیسه را دو بار وزن کنی و دو عدد کمی متفاوت بگیری. به همین لرزش می‌گوییم نویز.',
      ),
    },
    {
      say: bi(
        'A very big network can pass through every dot, noise included. Training loss almost zero!',
        'یک شبکه خیلی بزرگ می‌تواند از همه نقطه‌ها رد شود، حتی از نویزها. خطای آموزش تقریباً صفر!',
      ),
      more: bi(
        `This network has ${SIZES.knobs} knobs for only ${SIZES.dots} dots, so it has enough freedom to bend through every single dot. The training loss falls to almost 0. It looks perfect, but look at the shape: the green curve wiggles to catch every bit of noise.`,
        `این شبکه ${SIZES.knobs} پیچ دارد، فقط برای ${SIZES.dots} نقطه، پس آن‌قدر آزادی دارد که از تک‌تک نقطه‌ها رد شود. خطای آموزش تقریباً به ۰ می‌رسد. عالی به نظر می‌رسد، ولی به شکلش نگاه کن: منحنی سبز پیچ‌وتاب می‌خورد تا هر ذره نویز را بگیرد.`,
      ),
    },
    {
      say: bi(
        "But on new data it's bad: it memorised instead of learning. This is overfitting.",
        'ولی روی داده‌های جدید بد است: حفظ کرده، یاد نگرفته. به این «بیش‌برازش» (overfitting) می‌گوییم.',
      ),
      more: bi(
        `The hollow dots are new measurements from the same true rule that the network never saw. The red dashes show how far the curve is from each of them. On new dots the loss is much bigger than on the training dots. The network memorised the noise of its ${SIZES.dots} dots instead of learning the rule.`,
        `نقطه‌های توخالی اندازه‌گیری‌های جدیدی از همان قاعده واقعی‌اند که شبکه هیچ‌وقت ندیده. خط‌چین‌های قرمز نشان می‌دهند منحنی چقدر از هر کدام دور است. روی نقطه‌های جدید خطا خیلی بیشتر از نقطه‌های آموزش است. شبکه نویزِ ${SIZES.dots} نقطه‌اش را حفظ کرده، به جای اینکه قاعده را یاد بگیرد.`,
      ),
    },
    {
      say: bi(
        "That's why we set some data aside and only use it to check. When its loss starts rising, stop training.",
        'برای همین بخشی از داده را کنار می‌گذاریم و فقط با آن می‌سنجیم. وقتی خطای آن بالا رفت، آموزش را متوقف کن.',
      ),
      more: bi(
        'The red line is the loss on the training dots: it keeps falling. The teal line is the loss on the dots set aside to check: it falls at first, then turns and rises, because from there on the network starts learning the noise. The best moment is its lowest point. We keep the knobs from that step, and the curve on the right is calmer.',
        'خط قرمز خطا روی نقطه‌های آموزش است: مدام پایین می‌آید. خط فیروزه‌ای خطا روی نقطه‌هایی است که برای سنجیدن کنار گذاشتیم: اول پایین می‌آید، بعد برمی‌گردد و بالا می‌رود، چون از آن‌جا به بعد شبکه شروع می‌کند نویز را یاد بگیرد. بهترین لحظه پایین‌ترین نقطه آن است. پیچ‌های همان قدم را نگه می‌داریم و منحنی سمت دیگر آرام‌تر است.',
      ),
    },
    {
      say: bi(
        'When training is done, the knobs are locked. We just give inputs and get answers: inference.',
        'وقتی آموزش تمام شد، پیچ‌ها قفل می‌شوند. فقط ورودی می‌دهیم و جواب می‌گیریم: این «استفاده» (inference) است.',
      ),
      recall: { lesson: 'machine', note: bi('a machine: a number in, a number out', 'یک ماشین: عدد می‌دهی، عدد می‌گیری') },
      more: bi(
        'This is the number machine from lesson 1 again, now with a rule it learned by itself. The knobs are locked: they no longer change. A new number goes in, an answer comes out, and that is all. No target, no loss, no derivative: those were only needed while learning.',
        'این دوباره همان ماشین عدد درس ۱ است، این بار با قاعده‌ای که خودش یاد گرفته. پیچ‌ها قفل‌اند: دیگر عوض نمی‌شوند. یک عدد جدید وارد می‌شود، یک جواب بیرون می‌آید، و تمام. نه جواب درست، نه خطا، نه مشتق: این‌ها فقط موقع یاد گرفتن لازم بودند.',
      ),
    },
    {
      say: bi(
        'Look back at the whole road: machine, knobs, network, loss, derivative, walking downhill, gradient, chain, backpropagation, training. Every neural network, small or huge, runs on these few ideas.',
        'کل مسیر را مرور کن: ماشین، پیچ، شبکه، خطا، مشتق، پایین رفتن، گرادیان، زنجیره، پس‌انتشار، آموزش. همه شبکه‌های عصبی، کوچک یا بزرگ، با همین چند ایده کار می‌کنند.',
      ),
      more: bi(
        'A machine turns a number into a number. Its knobs set its rule. The loss says how wrong it is, the derivative says which way to turn each knob, and backpropagation finds all those derivatives quickly with the chain rule. Then we walk downhill, batch by batch, with good steps and a good start, and stop before it memorises. Huge networks do exactly this, only with many more knobs.',
        'یک ماشین عدد را به عدد تبدیل می‌کند. پیچ‌هایش قاعده‌اش را تعیین می‌کنند. خطا می‌گوید چقدر اشتباه کرده، مشتق می‌گوید هر پیچ را به کدام طرف بچرخانیم، و پس‌انتشار با قاعده زنجیره‌ای همه این مشتق‌ها را سریع پیدا می‌کند. بعد دسته به دسته، با قدم‌های خوب و شروع درست، سرازیر می‌شویم و قبل از اینکه حفظ کند متوقف می‌شویم. شبکه‌های خیلی بزرگ هم دقیقاً همین کار را می‌کنند، فقط با پیچ‌های خیلی بیشتر.',
      ),
    },
  ],
};
