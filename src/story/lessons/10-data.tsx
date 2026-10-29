// Lesson 10 · Learning from data: batches and epochs. Many examples, one direction from their
// average, a batch as a quick taste, epochs, and the bumpy loss of training.
// Script: docs/storyboard.md, lesson 10. Numbers: math-data.ts (engine backprop and trainStep).

import { useMemo, type ReactNode } from 'react';
import { bi, useT } from '../../i18n';
import { Plot2D, type PlotLayer } from '../../components/Plot2D';
import type { StoryLesson } from '../types';
import { Appear, C, Knob, Meter } from '../kit';
import { useTicker, useTweenArray } from '../useTween';
import { dataStory, lossContour, type DataStory } from './math-data';

const ID = 'data';
const EPOCHS = 8;
const PICK = 'var(--accent)';
const PATH = 'var(--ink-2)';
const LEVELS = [0.3, 1.2, 2.7, 4.8, 6.2, 7.6];
const H = 300;
/** Knob map view: zoomed in on the start for the arrows, zoomed out for training. */
const NEAR = [-0.2, 0.75, -0.25, 0.4];
const FAR = [-0.4, 2.6, -0.7, 1.6];

/** Steps of the scene (0-based), named after what they show. */
const S = { many: 0, average: 1, batch: 2, taste: 3, epoch: 4, train: 5, loss: 6 } as const;

function Panel({ title, children, width }: { title: ReactNode; children: ReactNode; width: number }) {
  return (
    <div style={{ flex: `1 1 ${width}px`, maxWidth: width, minWidth: 0, transition: 'max-width 0.6s ease, flex-basis 0.6s ease', display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ fontWeight: 700, fontSize: '1.02em', textAlign: 'center', color: 'var(--ink-2)' }}>{title}</div>
      {children}
    </div>
  );
}

/** The examples and the one-neuron line ŷ = w·x + b through them. */
function DataPlot({ s, w, b, picked, pulse }: { s: DataStory; w: number; b: number; picked: number[]; pulse: number }) {
  const layers: PlotLayer[] = [
    { kind: 'fn', f: (x) => w * x + b, color: C.output, width: 3.5 },
    {
      kind: 'points',
      color: PICK,
      pts: s.data.X.flatMap((x, n) => (picked.includes(n) || n === pulse ? [{ x: x[0], y: s.data.Y[n][0], r: 12, hollow: true, color: picked.includes(n) ? PICK : C.slope }] : [])),
    },
    { kind: 'points', color: C.target, pts: s.data.X.map((x, n) => ({ x: x[0], y: s.data.Y[n][0], r: 7 })) },
  ];
  return <Plot2D xDomain={[-2, 2.5]} yDomain={[-3, 6]} layers={layers} height={H} xLabel="x" yLabel="y" testId="data-plot" />;
}

const arrow = (from: [number, number], d: [number, number], color: string, width: number, opacity = 1, dash?: string): PlotLayer => ({
  kind: 'segment',
  from,
  to: [from[0] + d[0], from[1] + d[1]],
  color,
  width,
  opacity,
  dash,
  arrow: true,
});

/** The knob map: w across, b up, the loss on all examples drawn as contour rings. */
function KnobMap({ s, step, phase, iter, w, b }: { s: DataStory; step: number; phase: number; iter: number; w: number; b: number }) {
  const layers = useMemo<PlotLayer[]>(() => {
    const out: PlotLayer[] = LEVELS.map((L) => ({ kind: 'polyline', pts: lossContour(s.data, L), color: 'var(--line-strong)', width: 1.2 }));
    out.push({ kind: 'points', color: C.target, pts: [{ x: s.best.w, y: s.best.b, r: 6, hollow: true }] });
    return out;
  }, [s]);
  const start: [number, number] = [s.start.w, s.start.b];
  const [x0, x1, y0, y1] = useTweenArray(step >= S.average && step <= S.taste ? NEAR : FAR, 800);
  const dyn: PlotLayer[] = [];
  if (step === S.average) {
    const shown = Math.min(s.arrows.length, phase + 1);
    s.arrows.slice(0, shown).forEach((a, i) => dyn.push(arrow(start, a.step, C.slope, 2, phase >= 8 ? 0.3 : i === phase ? 1 : 0.55)));
    if (phase >= 8) dyn.push(arrow(start, s.mean, C.slope, 4));
  }
  if (step === S.batch) s.firstBatch.forEach((n) => dyn.push(arrow(start, s.arrows[n].step, C.slope, 2.5, 0.85)));
  if (step === S.taste) {
    dyn.push(arrow(start, s.mean, C.slope, 5, 0.35, '7 5'));
    dyn.push(arrow(start, s.batchMean, C.slope, 5));
  }
  if (step >= S.epoch && iter > 0) dyn.push({ kind: 'polyline', pts: s.frames.slice(0, iter + 1).map((f) => [f.w, f.b]), color: PATH, width: 1.6, markers: true, opacity: 0.7 });
  dyn.push({ kind: 'points', color: 'var(--ink)', pts: [{ x: w, y: b, r: 7 }] });
  return <Plot2D xDomain={[x0, x1]} yDomain={[y0, y1]} layers={[...layers, ...dyn]} height={H} xLabel="w" yLabel="b" testId="knob-map" />;
}

/** The loss of each batch during training, drawn up to update `upto`. */
function LossPlot({ s, upto }: { s: DataStory; upto: number }) {
  const { tr } = useT();
  const pts = s.frames.slice(1, upto + 1).map((f) => [f.iter, f.batchLoss] as [number, number]);
  const all = s.frames.slice(0, upto + 1).map((f) => [f.iter, f.loss] as [number, number]);
  const top = Math.max(...s.frames.slice(1).map((f) => f.batchLoss));
  const epochs = Array.from({ length: Math.floor((s.frames.length - 1) / s.perEpoch) - 1 }, (_, k) => (k + 1) * s.perEpoch + 0.5);
  const layers: PlotLayer[] = [
    ...epochs.map((at): PlotLayer => ({ kind: 'vline', at, color: 'var(--line-strong)', dash: '3 5' })),
    { kind: 'polyline', pts: all, color: 'var(--ink-2)', width: 1.6, dash: '6 4' },
    { kind: 'polyline', pts, color: C.loss, width: 2.5, markers: true },
  ];
  return (
    <Plot2D
      xDomain={[0, s.frames.length - 1]}
      yDomain={[0, top * 1.08]}
      layers={layers}
      height={H - 22}
      xLabel={tr(bi('step', 'قدم'))}
      yLabel="L"
      legend={[
        { label: tr(bi('loss of the batch', 'خطای دسته')), color: C.loss },
        { label: tr(bi('loss on all examples', 'خطا روی همه')), color: 'var(--ink-2)', dash: true },
      ]}
      testId="loss-plot"
    />
  );
}

/** What the arrows on the knob map mean (kept outside the plot so Persian text renders well). */
function Hint({ step, phase }: { step: number; phase: number }) {
  const { tr } = useT();
  const line = (style: string, text: string) => (
    <span key={style + text} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      <i style={{ display: 'inline-block', width: 26, borderTop: `${style === 'thick' ? 5 : 2}px ${style === 'dash' ? 'dashed' : 'solid'} ${C.slope}`, opacity: style === 'dash' ? 0.5 : 1 }} />
      {text}
    </span>
  );
  let items: ReactNode[] = [];
  if (step === S.average) items = [line('thin', tr(bi('one example', 'یک نمونه'))), phase >= 8 && line('thick', tr(bi('average of all 8', 'میانگین هر ۸')))];
  if (step === S.batch) items = [line('thin', tr(bi('the 2 batch examples', '۲ نمونه دسته')))];
  if (step === S.taste) items = [line('thick', tr(bi('average of the batch', 'میانگین دسته'))), line('dash', tr(bi('average of all 8', 'میانگین هر ۸')))];
  return <div style={{ display: 'flex', gap: 18, justifyContent: 'center', minHeight: 22, color: C.slope, fontWeight: 600, fontSize: '0.95em' }}>{items}</div>;
}

function Chips({ s, epoch, now, done }: { s: DataStory; epoch: number; now: number; done: boolean }) {
  const { tr } = useT();
  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'center', justifyContent: 'center', flexWrap: 'wrap' }} data-testid="batch-chips">
      <span style={{ fontWeight: 700, fontSize: '1.15em', minWidth: 90, textAlign: 'center', color: done ? C.output : 'var(--ink)' }}>
        {tr(bi('epoch', 'دوره'))} <span className="num">{epoch}</span> {done ? '✓' : ''}
      </span>
      {s.epoch1.map((_, k) => {
        const lit = k + 1 === now;
        const past = k + 1 < now || done;
        return (
          <span
            key={k}
            style={{
              padding: '6px 14px',
              borderRadius: 999,
              border: `2px solid ${lit ? PICK : past ? C.output : 'var(--line-strong)'}`,
              background: lit ? 'var(--accent-soft)' : 'var(--surface)',
              fontWeight: lit ? 700 : 500,
              transition: 'all 0.3s',
              opacity: past || lit ? 1 : 0.55,
            }}
          >
            {tr(bi('batch', 'دسته'))} {k + 1}
          </span>
        );
      })}
    </div>
  );
}

function Scene({ step }: { step: number }) {
  const { tr } = useT();
  const s = useMemo(() => dataStory(EPOCHS), []);
  const last = s.frames.length - 1;
  const phase = useTicker(step === S.average, 8, 420);
  const tEpoch = useTicker(step === S.epoch, s.perEpoch, 1100);
  const tTrain = useTicker(step === S.train, last - s.perEpoch, 220);
  const tLoss = useTicker(step === S.loss, last, 45);
  const iter = step < S.epoch ? 0 : step === S.epoch ? tEpoch : step === S.train ? s.perEpoch + tTrain : last;
  const f = s.frames[iter];
  const [w, b] = useTweenArray([f.w, f.b], step === S.train ? 210 : 650);
  const picked = step === S.batch || step === S.taste ? s.firstBatch : (step === S.epoch || step === S.train) && iter > 0 ? f.batch : [];
  const pulse = step === S.average && phase < 8 ? phase : -1;
  const epoch = Math.max(1, Math.ceil(iter / s.perEpoch));
  const now = iter === 0 ? 0 : ((iter - 1) % s.perEpoch) + 1;
  const wide = step === S.many;

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
      {step >= S.epoch && <Chips s={s} epoch={epoch} now={step === S.loss ? 0 : now} done={step === S.loss || (step === S.epoch && tEpoch >= s.perEpoch) || (step === S.train && tTrain >= last - s.perEpoch)} />}
      <div style={{ display: 'flex', gap: 24, width: '100%', justifyContent: 'center', alignItems: 'flex-start' }}>
        <Panel title={tr(bi(`${s.data.X.length} examples and the line ŷ = w·x + b`, `${s.data.X.length} نمونه و خط ŷ = w·x + b`))} width={wide ? 720 : 560}>
          <DataPlot s={s} w={w} b={b} picked={picked} pulse={pulse} />
        </Panel>
        {!wide && (
          <Panel title={step === S.loss ? tr(bi(`loss during training: ${EPOCHS} epochs × ${s.perEpoch} batches`, `خطا در طول آموزش: ${EPOCHS} دوره × ${s.perEpoch} دسته`)) : tr(bi('the knobs w and b (rings: equal loss)', 'پیچ‌های w و b (حلقه‌ها: خطای برابر)'))} width={560}>
            <Hint step={step} phase={phase} />
            <Appear show>{step === S.loss ? <LossPlot s={s} upto={tLoss} /> : <KnobMap s={s} step={step} phase={phase} iter={iter} w={w} b={b} />}</Appear>
          </Panel>
        )}
      </div>
      <div style={{ display: 'flex', gap: 28, alignItems: 'center', justifyContent: 'center' }}>
        <Knob label="w" value={f.w} min={-0.5} max={3} size={78} />
        <Knob label="b" value={f.b} min={-1} max={2} color={C.bias} size={78} />
        <Meter value={f.loss} max={8} label={tr(bi('loss on all examples', 'خطا روی همه نمونه‌ها'))} digits={2} />
      </div>
    </div>
  );
}

export const dataLesson: StoryLesson = {
  id: ID,
  presetId: 'mini-batch',
  Scene,
  steps: [
    {
      say: bi('So far we had one example. In reality we have thousands.', 'تا اینجا فقط یک نمونه داشتیم. در واقعیت هزاران نمونه داریم.'),
      more: bi(
        'Each gold dot is one example: an input x and its real answer y. Here we draw only 8 so you can see each one, but real data can have thousands. The green line is our one-neuron machine $\\hat{y} = w x + b$, still with its starting knobs, so it fits none of them yet.',
        'هر نقطه طلایی یک نمونه است: یک ورودی x و جواب واقعی‌اش y. این‌جا فقط ۸ تا کشیده‌ایم تا هر کدام را ببینی، ولی داده واقعی می‌تواند هزاران نمونه داشته باشد. خط سبز همان ماشین تک‌نورونی $\\hat{y} = w x + b$ است، هنوز با پیچ‌های اولش، برای همین فعلاً از هیچ‌کدام رد نمی‌شود.',
      ),
    },
    {
      say: bi(
        'Compute the derivative for each example and average them: a direction that is good for all.',
        'مشتق را برای هر نمونه حساب کن و میانگین بگیر: جهتی که برای همه خوب است.',
      ),
      more: bi(
        'On the right is the map of the two knobs: w across, b up. Each dot, one by one, gives its own downhill arrow: where it would move the knobs if it were alone. The arrows disagree, so we add them up and divide by 8. That thick average arrow is the step for all examples together.',
        'سمت راست نقشه دو پیچ است: w به راست، b به بالا. هر نقطه، یکی‌یکی، پیکان سرازیری خودش را می‌دهد: اگر تنها بود پیچ‌ها را به کجا می‌برد. پیکان‌ها با هم فرق دارند، پس جمعشان می‌کنیم و بر ۸ تقسیم می‌کنیم. آن پیکان کلفت میانگین، قدمی است که برای همه نمونه‌ها با هم خوب است.',
      ),
    },
    {
      say: bi(
        'But using every example for each step is slow. Instead we take a few: a batch.',
        'ولی حساب همه نمونه‌ها برای هر قدم کند است. به جایش چند نمونه برمی‌داریم: یک «دسته» (batch).',
      ),
      more: bi(
        'With thousands of examples, computing every arrow before each single step would take a long time. So we pick a few examples, here 2, and use only their arrows. The two circled dots are this batch; only their two arrows are left on the map.',
        'با هزاران نمونه، حساب کردن همه پیکان‌ها قبل از هر قدم خیلی طول می‌کشد. پس چند نمونه برمی‌داریم، این‌جا ۲ تا، و فقط پیکان‌های همان‌ها را حساب می‌کنیم. دو نقطه حلقه‌دار همین دسته‌اند؛ روی نقشه فقط دو پیکان آن‌ها مانده.',
      ),
    },
    {
      say: bi(
        "A batch's direction is a little off but roughly right, like tasting one spoonful instead of the whole pot.",
        'جهتِ یک دسته کمی کج است ولی تقریباً درست، مثل چشیدن یک قاشق سوپ به جای کل قابلمه.',
      ),
      more: bi(
        'The solid arrow is the average of only the 2 batch examples. The faded dashed arrow is the average of all 8. They point almost the same way, but not exactly. One spoonful tells you if the soup needs salt, even if it is not the whole pot.',
        'پیکان پررنگ میانگین فقط ۲ نمونه دسته است. پیکان کم‌رنگ خط‌چین میانگین هر ۸ نمونه است. تقریباً به یک طرف می‌روند، ولی نه دقیقاً. یک قاشق هم می‌گوید سوپ نمک لازم دارد یا نه، با اینکه کل قابلمه نیست.',
      ),
    },
    {
      say: bi('When every batch has been seen once, one epoch is done.', 'وقتی همه دسته‌ها یک بار دیده شدند، یک «دوره» (epoch) تمام شده.'),
      more: bi(
        'The 8 examples are shuffled and cut into 4 batches of 2. Each batch gives one step: its dots are circled, the knobs move, and the line turns a little. After batch 4 every example has been used exactly once: that is one epoch. Watch the knob dot walk on the map, one step per batch.',
        '۸ نمونه را بُر می‌زنیم و به ۴ دسته ۲تایی تقسیم می‌کنیم. هر دسته یک قدم می‌دهد: نقطه‌هایش حلقه‌دار می‌شوند، پیچ‌ها تکان می‌خورند و خط کمی می‌چرخد. بعد از دسته ۴، هر نمونه دقیقاً یک بار استفاده شده: این یعنی یک دوره. ببین نقطه پیچ‌ها روی نقشه چطور با هر دسته یک قدم برمی‌دارد.',
      ),
    },
    {
      say: bi('After a few epochs the line settles onto the data.', 'چند دوره که بگذرد، خط روی داده‌ها می‌نشیند.'),
      more: bi(
        'Each new epoch shuffles the examples again, so the batches change. Step after step the knobs walk to the bottom of the rings, and the green line settles in the middle of the dots. It cannot touch every dot, because the dots are not on one perfect line.',
        'هر دوره جدید نمونه‌ها را دوباره بُر می‌زند، پس دسته‌ها عوض می‌شوند. قدم به قدم پیچ‌ها به ته حلقه‌ها می‌رسند و خط سبز وسط نقطه‌ها می‌نشیند. نمی‌تواند از همه نقطه‌ها رد شود، چون نقطه‌ها دقیقاً روی یک خط نیستند.',
      ),
    },
    {
      say: bi(
        'The loss falls during training, a little bumpy because every batch is different.',
        'نمودار خطا پایین می‌آید، کمی لرزان، چون هر دسته فرق دارد.',
      ),
      more: bi(
        'Each red dot is the loss of the batch used in that step. Overall it falls fast, but not smoothly: some batches are easier than others, so it goes up and down. The dashed line is the loss on all 8 examples: it falls more calmly. A bumpy curve like the red one is what you see when a real network trains.',
        'هر نقطه قرمز خطای دسته‌ای است که در آن قدم استفاده شد. در کل سریع پایین می‌آید، ولی صاف نه: بعضی دسته‌ها از بقیه آسان‌ترند، پس بالا و پایین می‌رود. خط‌چین خطا روی هر ۸ نمونه است: آرام‌تر پایین می‌آید. منحنی لرزانی مثل همین قرمز چیزی است که موقع آموزش یک شبکه واقعی می‌بینی.',
      ),
    },
  ],
};
