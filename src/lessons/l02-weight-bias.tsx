// Lesson 02: Weight and bias. z = w·x + b for one input, z = w1·x1 + w2·x2 + b for two, and the
// precise difference between linear and affine maps. Lesson-local state; z comes from the engine.

import { useMemo } from 'react';
import type { LessonDef } from './types';
import { bi, useT } from '../i18n';
import { useLessonState } from '../state/lessonState';
import { Plot2D, type PlotLayer } from '../components/Plot2D';
import { Tex } from '../components/Tex';
import { Segmented, Slider, Stat, type MathColor } from '../components/controls';
import { Advanced, Callout, Cycle, P, Predict, Reveal } from '../components/teaching';
import { clamp, fmt, fmtSigned, niceTicks, predict, tex, texP, type Network } from '../engine';

const ID = 'l02';

type Mode = 'one' | 'two' | 'linear';
interface S {
  mode: Mode;
  x: number;
  w: number;
  b: number;
  x1: number;
  x2: number;
  w1: number;
  w2: number;
  u: number;
  v: number;
  al: number;
  be: number;
}
const INIT: S = { mode: 'one', x: 1.5, w: 2, b: 1, x1: 2, x2: 1, w1: 1.5, w2: -1, u: 1, v: 2, al: 2, be: 1 };
const useS = () => useLessonState<S>(ID, INIT);

/** A single linear neuron with the given weights, so z always comes from the engine's forward pass. */
const neuron = (ws: number[], b: number): Network => ({ inputSize: ws.length, layers: [{ W: [ws], b: [b], act: 'linear' }] });
const zOf = (ws: number[], b: number, xs: number[]) => predict(neuron(ws, b), xs)[0];
/** Hide floating-point dust (1e-16) in differences that are exactly zero in exact arithmetic. */
const clean = (v: number) => (Math.abs(v) < 1e-9 ? 0 : v);

function Explain() {
  const { tr } = useT();
  const [s, set] = useS();
  return (
    <>
      <Cycle active={s.mode === 'linear' ? 3 : 1} />
      <Reveal at={1}>
        <h3>{tr(bi('1. A neuron\'s first step', '۱. اولین کار یک نورون'))}</h3>
        <P>
          {bi(
            'A neuron starts with a tiny calculation: multiply the input by a weight, then add a bias.',
            'یک نورون کارش را با یک حساب خیلی ساده شروع می‌کند: ورودی را در یک وزن (weight) ضرب می‌کند و بعد یک بایاس (bias) به آن اضافه می‌کند.',
          )}
        </P>
        <Tex display>{'\\cz{z} = \\cw{w}\\,\\cx{x} + \\cb{b}'}</Tex>
        <P>
          {bi(
            'Think of a taxi fare: price $=$ rate $\\times$ kilometres $+$ fixed fee. The rate is the weight (how much each unit of input counts), the fixed fee is the bias (what you pay even for zero kilometres). $\\cw{w}$ and $\\cb{b}$ are the knobs a network learns.',
            'کرایه تاکسی را در نظر بگیر: کرایه $=$ نرخ $\\times$ کیلومتر $+$ ورودیه ثابت. نرخ همان وزن است (هر واحد ورودی چقدر حساب می‌شود) و ورودیه ثابت همان بایاس (چیزی که حتی برای صفر کیلومتر می‌پردازی). $\\cw{w}$ و $\\cb{b}$ پیچ‌هایی هستند که شبکه یاد می‌گیرد.',
          )}
        </P>
      </Reveal>
      <Reveal at={2}>
        <h3>{tr(bi('2. The weight sets the slope', '۲. وزن شیب را تعیین می‌کند'))}</h3>
        <P>
          {bi(
            'Each step of $+1$ in $\\cx{x}$ changes $\\cz{z}$ by exactly $\\cw{w}$. A large $|w|$ makes the neuron very sensitive to its input (a steep line). $w = 0$ means the input is ignored (a flat line).',
            'هر قدم $+1$ در $\\cx{x}$ مقدار $\\cz{z}$ را دقیقاً به اندازه $\\cw{w}$ عوض می‌کند. اگر $|w|$ بزرگ باشد نورون به ورودی‌اش خیلی حساس است (خط تند). $w = 0$ یعنی ورودی نادیده گرفته می‌شود (خط افقی).',
          )}
        </P>
        <Predict
          id={ID + '-neg'}
          question={bi('Drag $w$ from $+2$ to $-2$ (keep $b$). What happens to the line?', 'اگر $w$ را از $+2$ به $-2$ ببری (و $b$ را نگه داری)، خط چه می‌شود؟')}
          choices={[
            { id: 'flip', label: bi('It tilts the other way: bigger x gives smaller z', 'به سمت دیگر کج می‌شود: x بزرگ‌تر، z کوچک‌تر') },
            { id: 'down', label: bi('It moves down but keeps its direction', 'پایین می‌رود ولی جهتش همان می‌ماند') },
            { id: 'same', label: bi('Nothing: only the size of w matters', 'هیچ: فقط اندازه w مهم است') },
          ]}
          answer="flip"
          explain={bi(
            'A negative weight reverses the input\'s effect: going right now goes down. The steepness is the same because $|w|$ is still 2, and the line still crosses $x = 0$ at $b$. On the network diagram negative weights are drawn orange, positive ones teal.',
            'وزن منفی اثر ورودی را برعکس می‌کند: حالا با رفتن به راست پایین می‌آیی. تندی خط همان است چون $|w|$ هنوز ۲ است، و خط هنوز در $x = 0$ از $b$ می‌گذرد. در نمودار شبکه وزن‌های منفی نارنجی و مثبت‌ها سبزآبی کشیده می‌شوند.',
          )}
        />
      </Reveal>
      <Reveal at={3}>
        <h3>{tr(bi('3. The bias shifts the line', '۳. بایاس خط را جابه‌جا می‌کند'))}</h3>
        <P>
          {bi(
            'At $x = 0$ the weight term vanishes and $z = b$: the bias is where the line crosses the vertical axis. Changing $b$ slides the whole line up or down without changing its slope. It also moves the point where $z$ crosses zero, $x = -b/w$. That crossing becomes very important once we add ReLU in the next lesson.',
            'در $x = 0$ جمله وزن صفر می‌شود و $z = b$: بایاس همان جایی است که خط محور عمودی را قطع می‌کند. با تغییر $b$ کل خط بالا یا پایین می‌رود و شیبش عوض نمی‌شود. نقطه‌ای که $z$ از صفر رد می‌شود، یعنی $x = -b/w$، هم جابه‌جا می‌شود. این نقطه در درس بعد، وقتی ReLU اضافه شود، خیلی مهم می‌شود.',
          )}
        </P>
      </Reveal>
      <Reveal at={4}>
        <h3>{tr(bi('4. Several inputs: add up the contributions', '۴. چند ورودی: سهم‌ها جمع می‌شوند'))}</h3>
        <P>
          {bi(
            'Real neurons read several inputs. Each input gets its own weight, each product is that input\'s contribution, and the neuron simply adds them all plus one bias:',
            'نورون‌های واقعی چند ورودی می‌خوانند. هر ورودی وزن خودش را دارد، هر حاصل‌ضرب سهم آن ورودی است، و نورون همه را با هم و با یک بایاس جمع می‌کند:',
          )}
        </P>
        <Tex display>{'\\cz{z} = \\cw{w_1}\\cx{x_1} + \\cw{w_2}\\cx{x_2} + \\cb{b}'}</Tex>
        <P>
          {bi(
            'Why a sum? Because it is the simplest way to let every input push $z$ up or down independently: a positive contribution pushes up, a negative one pulls down. With one input the graph is a line; with two it is a tilted flat plane. Seen from above, the points with equal $z$ form parallel straight stripes.',
            'چرا جمع؟ چون ساده‌ترین راه است که هر ورودی بتواند مستقل از بقیه $z$ را بالا یا پایین ببرد: سهم مثبت بالا می‌برد و سهم منفی پایین می‌کشد. با یک ورودی نمودار یک خط است؛ با دو ورودی یک صفحه صاف و کج. اگر از بالا نگاه کنی، نقطه‌هایی که $z$ برابر دارند نوارهای راست و موازی می‌سازند.',
          )}
        </P>
        {s.mode !== 'two' && (
          <button className="btn sm" onClick={() => set({ mode: 'two' })} data-testid="to-two">
            {tr(bi('Try two inputs →', 'دو ورودی را امتحان کن ←'))}
          </button>
        )}
      </Reveal>
      <Reveal at={5}>
        <h3>{tr(bi('5. Linear or affine?', '۵. خطی یا آفین؟'))}</h3>
        <P>
          {bi(
            'In everyday ML talk, $wx + b$ is called "linear". In math, a function is linear only if mixing inputs mixes outputs the same way: $f(\\alpha u + \\beta v) = \\alpha f(u) + \\beta f(v)$ for all numbers $\\alpha, \\beta$. That forces $f(0) = 0$: the graph passes through the origin.',
            'در صحبت روزمره یادگیری ماشین به $wx + b$ می‌گویند «خطی». در ریاضی، تابعی خطی (linear) است که ترکیب ورودی‌ها را به همان شکل در خروجی ترکیب کند: $f(\\alpha u + \\beta v) = \\alpha f(u) + \\beta f(v)$ برای همه عددهای $\\alpha$ و $\\beta$. نتیجه‌اش این است که $f(0) = 0$: نمودار از مبدأ می‌گذرد.',
          )}
        </P>
        <P>
          {bi(
            '$wx$ is linear. $wx + b$ with $b \\ne 0$ is affine: a linear map plus a constant. The two sides differ by exactly $b\\,(1 - \\alpha - \\beta)$. Open the test and check.',
            '$wx$ خطی است. $wx + b$ با $b \\ne 0$ آفین (affine) است: یک نگاشت خطی به‌علاوه یک مقدار ثابت. دو طرف دقیقاً به اندازه $b\\,(1 - \\alpha - \\beta)$ با هم فرق دارند. آزمون را باز کن و خودت ببین.',
          )}
        </P>
        {s.mode !== 'linear' && (
          <button className="btn sm" onClick={() => set({ mode: 'linear' })} data-testid="to-linear">
            {tr(bi('Open the linearity test →', 'آزمون خطی بودن را باز کن ←'))}
          </button>
        )}
      </Reveal>
      <Reveal at={6}>
        <Advanced>
          <P>
            {bi(
              'With $n$ inputs: $z = \\mathbf{w}\\cdot\\mathbf{x} + b = \\sum_{i=1}^n w_i x_i + b$. Appending a constant input $x_0 = 1$ with weight $w_0 = b$ turns it into a pure dot product (the "bias trick"), which is why an affine map in $n$ dimensions is a linear map in $n+1$.',
              'با $n$ ورودی: $z = \\mathbf{w}\\cdot\\mathbf{x} + b = \\sum_{i=1}^n w_i x_i + b$. اگر یک ورودی ثابت $x_0 = 1$ با وزن $w_0 = b$ اضافه کنیم، فقط یک ضرب داخلی باقی می‌ماند («ترفند بایاس»)؛ برای همین یک نگاشت آفین در $n$ بُعد همان یک نگاشت خطی در $n+1$ بُعد است.',
            )}
          </P>
          <P>
            {bi(
              'Affine maps preserve affine combinations: if $\\alpha + \\beta = 1$ then $f(\\alpha u + \\beta v) = \\alpha f(u) + \\beta f(v)$ even with a bias. The gradient $\\nabla_{\\mathbf{x}} z = \\mathbf{w}$ points in the direction of fastest increase and is perpendicular to the level sets $\\{\\mathbf{x} : z = c\\}$, which are hyperplanes. The set $z = 0$ is the decision boundary of a linear classifier.',
              'نگاشت‌های آفین ترکیب‌های آفین را حفظ می‌کنند: اگر $\\alpha + \\beta = 1$ باشد، حتی با بایاس هم $f(\\alpha u + \\beta v) = \\alpha f(u) + \\beta f(v)$. گرادیان $\\nabla_{\\mathbf{x}} z = \\mathbf{w}$ جهت بیشترین افزایش را نشان می‌دهد و بر مجموعه‌های هم‌تراز $\\{\\mathbf{x} : z = c\\}$ عمود است که ابرصفحه (hyperplane) هستند. مجموعه $z = 0$ همان مرز تصمیم یک طبقه‌بند خطی است.',
            )}
          </P>
          <P>
            {bi(
              'PyTorch\'s nn.Linear computes $W\\mathbf{x} + \\mathbf{b}$, an affine map, despite its name.',
              'لایه nn.Linear در PyTorch با وجود اسمش $W\\mathbf{x} + \\mathbf{b}$ را حساب می‌کند که یک نگاشت آفین است.',
            )}
          </P>
        </Advanced>
      </Reveal>
    </>
  );
}

/** Inputs on the left, weighted edges (teal = positive, orange = negative, width ~ |w|), bias from below. */
function NeuronDiagram({ inputs, b, z }: { inputs: { name: string; v: number; w: number; wName: string }[]; b: number; z: number }) {
  const H = inputs.length === 1 ? 150 : 200;
  const nx = 260;
  const ny = inputs.length === 1 ? 64 : 92;
  const maxW = Math.max(1, ...inputs.map((i) => Math.abs(i.w)));
  return (
    <svg viewBox={`0 0 460 ${H}`} width="100%" style={{ maxWidth: 520, display: 'block', margin: '0 auto', direction: 'ltr' }} role="img" aria-label="neuron diagram" data-testid="neuron-diagram">
      {inputs.map((inp, k) => {
        const iy = inputs.length === 1 ? ny : 46 + k * 92;
        const ix = 56;
        const stroke = inp.w >= 0 ? 'var(--c-wpos)' : 'var(--c-wneg)';
        return (
          <g key={k}>
            <line x1={ix + 24} y1={iy} x2={nx - 32} y2={ny} stroke={stroke} strokeWidth={1 + (4 * Math.abs(inp.w)) / maxW} opacity={inp.w === 0 ? 0.25 : 1} />
            <text x={(ix + nx) / 2 - 4} y={(iy + ny) / 2 - 10} textAnchor="middle" fill="var(--c-w)" fontSize={13}>
              {inp.wName} = {fmt(inp.w, 2)}
            </text>
            <circle cx={ix} cy={iy} r={24} fill="var(--surface)" stroke="var(--c-x)" strokeWidth={2} />
            <text x={ix} y={iy + 5} textAnchor="middle" fill="var(--c-x)" fontSize={13} fontFamily="var(--font-mono)">
              {fmt(inp.v, 2)}
            </text>
            <text x={ix} y={iy - 30} textAnchor="middle" fill="var(--c-x)" fontSize={12}>
              {inp.name}
            </text>
          </g>
        );
      })}
      <line x1={nx} y1={H - 14} x2={nx} y2={ny + 32} stroke="var(--c-b)" strokeWidth={2} />
      <text x={nx + 8} y={H - 8} fill="var(--c-b)" fontSize={13}>
        b = {fmt(b, 2)}
      </text>
      <circle cx={nx} cy={ny} r={32} fill="var(--surface-2)" stroke="var(--c-z)" strokeWidth={2} />
      <text x={nx} y={ny - 4} textAnchor="middle" fill="var(--ink)" fontSize={18}>
        Σ
      </text>
      <text x={nx} y={ny + 16} textAnchor="middle" fill="var(--c-z)" fontSize={12} fontFamily="var(--font-mono)">
        +b
      </text>
      <line x1={nx + 32} y1={ny} x2={400} y2={ny} stroke="var(--c-z)" strokeWidth={2} />
      <text x={406} y={ny + 5} fill="var(--c-z)" fontSize={14} fontFamily="var(--font-mono)">
        z = {fmt(z, 2)}
      </text>
    </svg>
  );
}

/** Each term as a bar that starts where the previous one ended; the last bar is the total. */
function Waterfall({ terms, total }: { terms: { label: string; v: number; color: MathColor }[]; total: number }) {
  let acc = 0;
  const spans = terms.map((t) => {
    const from = acc;
    acc += t.v;
    return { ...t, from, to: acc };
  });
  const ends = spans.flatMap((s) => [s.from, s.to]);
  const lo = Math.min(0, total, ...ends);
  const hi = Math.max(0, total, ...ends);
  const pad = (hi - lo) * 0.06 || 1;
  const pct = (v: number) => ((v - (lo - pad)) / (hi - lo + 2 * pad)) * 100;
  const row = (label: string, from: number, to: number, color: string, value: number, strong = false) => (
    <div key={label} style={{ display: 'grid', gridTemplateColumns: '7.5em 1fr 4.5em', alignItems: 'center', gap: 8 }}>
      <Tex>{label}</Tex>
      <div dir="ltr" style={{ position: 'relative', height: 18, background: 'var(--surface-2)', borderRadius: 4 }}>
        <div style={{ position: 'absolute', left: `${pct(0)}%`, top: -2, bottom: -2, borderLeft: '1px dashed var(--line-strong)' }} />
        <div
          style={{
            position: 'absolute',
            left: `${pct(Math.min(from, to))}%`,
            width: `${Math.max(0.4, pct(Math.max(from, to)) - pct(Math.min(from, to)))}%`,
            top: 3,
            bottom: 3,
            borderRadius: 3,
            background: `var(--c-${color})`,
            opacity: strong ? 1 : 0.8,
          }}
        />
      </div>
      <span className={`num c-${color}`} style={{ fontWeight: strong ? 700 : 400 }}>
        {fmtSigned(value, 2)}
      </span>
    </div>
  );
  return (
    <div className="stack" style={{ gap: 6 }} data-testid="waterfall">
      {spans.map((s) => row(s.label, s.from, s.to, s.color, s.v))}
      {row('\\cz{z}', 0, total, 'z', total, true)}
    </div>
  );
}

function linePlotLayers(s: S): PlotLayer[] {
  const z = zOf([s.w], s.b, [s.x]);
  const out: PlotLayer[] = [
    { kind: 'line', p: [0, 0], slope: 1, color: 'var(--line-strong)', dash: '4 5', width: 1.4 },
    { kind: 'fn', f: (t) => zOf([s.w], s.b, [t]), color: 'z', width: 2.8 },
    { kind: 'segment', from: [0, 0], to: [0, s.b], color: 'b', width: 3.2, label: 'b' },
    { kind: 'segment', from: [s.x, z], to: [s.x + 1, z], color: 'x', width: 1.8, dash: '3 3', label: '+1' },
    { kind: 'segment', from: [s.x + 1, z], to: [s.x + 1, z + s.w], color: 'w', width: 2.2, label: `w = ${fmt(s.w, 2)}` },
  ];
  if (Math.abs(s.w) > 1e-9) {
    const x0 = -s.b / s.w;
    out.push({ kind: 'points', color: 'z', pts: [{ x: x0, y: 0, r: 4.5, hollow: true, label: 'z = 0' }] });
  }
  return out;
}

function contourLayers(s: S): PlotLayer[] {
  const corners = [-4, 4].flatMap((a) => [-4, 4].map((c) => s.w1 * a + s.w2 * c + s.b));
  const levels = niceTicks(Math.min(...corners), Math.max(...corners), 8);
  const out: PlotLayer[] = [];
  for (const c of levels) {
    const main = Math.abs(c) < 1e-9;
    if (Math.abs(s.w2) > 1e-6) {
      const f = (x1: number) => (c - s.b - s.w1 * x1) / s.w2;
      out.push({ kind: 'fn', f, color: 'z', width: main ? 2.4 : 1.2, opacity: main ? 1 : 0.45, samples: 4 });
      const lx = [3.3, 2.3, 1.3, 0.3, -0.7, -1.7, -2.7, -3.7].find((t) => Math.abs(f(t)) < 3.5);
      if (lx !== undefined) out.push({ kind: 'text', x: lx, y: f(lx), text: `z=${fmt(c, 2)}`, color: 'z', dy: -4 });
    } else if (Math.abs(s.w1) > 1e-6) {
      out.push({ kind: 'vline', at: (c - s.b) / s.w1, color: 'z', dash: main ? '0' : '3 3', label: `z=${fmt(c, 2)}` });
    }
  }
  const n = Math.hypot(s.w1, s.w2);
  if (n > 1e-9) {
    out.push({ kind: 'segment', from: [s.x1, s.x2], to: [s.x1 + (1.3 * s.w1) / n, s.x2 + (1.3 * s.w2) / n], color: 'w', width: 2.4, arrow: true, label: '↑z' });
  }
  return out;
}

function linearLayers(s: S): PlotLayer[] {
  const f = (t: number) => zOf([s.w], s.b, [t]);
  const m = s.al * s.u + s.be * s.v;
  const lhs = f(m);
  const rhs = s.al * f(s.u) + s.be * f(s.v);
  return [
    { kind: 'line', p: [0, 0], slope: s.w, color: 'var(--line-strong)', dash: '5 4', width: 1.4 },
    { kind: 'fn', f, color: 'z', width: 2.6 },
    { kind: 'points', color: 'x', pts: [{ x: s.u, y: f(s.u), label: 'f(u)' }, { x: s.v, y: f(s.v), label: 'f(v)' }] },
    { kind: 'segment', from: [m, lhs], to: [m, rhs], color: 'b', width: 3, label: Math.abs(lhs - rhs) > 1e-9 ? fmtSigned(rhs - lhs, 2) : undefined },
    { kind: 'points', color: 'z', pts: [{ x: m, y: lhs, label: 'f(αu+βv)' }] },
    { kind: 'points', color: 'g', pts: [{ x: m, y: rhs, hollow: true, r: 6, label: 'αf(u)+βf(v)' }] },
  ];
}

function Center() {
  const { tr } = useT();
  const [s, set] = useS();
  const z = s.mode === 'two' ? zOf([s.w1, s.w2], s.b, [s.x1, s.x2]) : zOf([s.w], s.b, [s.x]);
  const layers = useMemo(() => (s.mode === 'one' ? linePlotLayers(s) : s.mode === 'two' ? contourLayers(s) : linearLayers(s)), [s]);
  return (
    <div className="viz">
      <div className="row">
        <Segmented
          label="mode"
          value={s.mode}
          onChange={(mode) => set({ mode })}
          options={[
            { value: 'one', label: tr(bi('One input', 'یک ورودی')) },
            { value: 'two', label: tr(bi('Two inputs', 'دو ورودی')) },
            { value: 'linear', label: tr(bi('Linear vs affine', 'خطی یا آفین')) },
          ]}
        />
      </div>
      {s.mode !== 'linear' && (
        <NeuronDiagram
          inputs={s.mode === 'one' ? [{ name: 'x', v: s.x, w: s.w, wName: 'w' }] : [{ name: 'x₁', v: s.x1, w: s.w1, wName: 'w₁' }, { name: 'x₂', v: s.x2, w: s.w2, wName: 'w₂' }]}
          b={s.b}
          z={z}
        />
      )}
      {s.mode === 'one' && (
        <Plot2D
          testId="affine-plot"
          xDomain={[-4, 4]}
          yDomain={[-10, 10]}
          layers={layers}
          height={300}
          xLabel={tr(bi('input x', 'ورودی x'))}
          yLabel="z = w·x + b"
          handle={{ x: s.x, y: z, color: 'z' }}
          onDrag={(x) => set({ x: clamp(Math.round(x * 10) / 10, -4, 4) })}
          legend={[
            { label: 'z = w·x + b', color: 'z' },
            { label: tr(bi('reference: w = 1, b = 0', 'مرجع: w = 1 و b = 0')), color: 'var(--line-strong)', dash: true },
          ]}
          ariaLabel="z as a function of x"
        />
      )}
      {s.mode === 'two' && (
        <>
          <Plot2D
            testId="contour-plot"
            xDomain={[-4, 4]}
            yDomain={[-4, 4]}
            layers={layers}
            height={300}
            xLabel="x₁"
            yLabel="x₂"
            handle={{ x: s.x1, y: s.x2, color: 'x' }}
            onDrag={(a, c) => set({ x1: clamp(Math.round(a * 10) / 10, -4, 4), x2: clamp(Math.round(c * 10) / 10, -4, 4) })}
            ariaLabel="level lines of z over the input plane"
          />
          <p className="small muted">
            {tr(
              bi(
                'Seen from above: each line joins inputs with the same z. Drag the point; the arrow shows the direction in which z grows fastest.',
                'از بالا: هر خط ورودی‌هایی را که z برابر دارند به هم وصل می‌کند. نقطه را بکش؛ فلش جهتی را نشان می‌دهد که z در آن سریع‌تر از همه زیاد می‌شود.',
              ),
            )}
          </p>
        </>
      )}
      {s.mode === 'linear' && (
        <Plot2D
          testId="linearity-plot"
          xDomain={[-4, 6]}
          yDomain="auto"
          layers={layers}
          height={340}
          xLabel="t"
          yLabel="f(t) = w·t + b"
          legend={[
            { label: 'f(t)', color: 'z' },
            { label: tr(bi('same slope, b = 0 (truly linear)', 'همان شیب با b = 0 (واقعاً خطی)')), color: 'var(--line-strong)', dash: true },
            { label: tr(bi('gap = b(1 − α − β)', 'فاصله = b(1 − α − β)')), color: 'b' },
          ]}
          ariaLabel="linearity test"
        />
      )}
      {s.mode === 'one' && <p className="small muted">{tr(bi('Drag on the plot to change x.', 'روی نمودار بکش تا x عوض شود.'))}</p>}
    </div>
  );
}

function slopeWords(w: number) {
  if (Math.abs(w) < 1e-9) return bi('w = 0: the input is ignored, the line is flat.', 'w = 0: ورودی نادیده گرفته می‌شود و خط افقی است.');
  if (w < 0) return bi('w < 0: bigger x gives smaller z (the line goes down to the right).', 'w < 0: x بزرگ‌تر، z کوچک‌تر (خط به سمت راست پایین می‌رود).');
  if (w > 1) return bi('w > 1: z changes faster than x (steeper than the reference).', 'w > 1: z سریع‌تر از x تغییر می‌کند (تندتر از خط مرجع).');
  if (w < 1) return bi('0 < w < 1: z changes slower than x (flatter than the reference).', '0 < w < 1: z کندتر از x تغییر می‌کند (کم‌شیب‌تر از خط مرجع).');
  return bi('w = 1: z changes exactly as fast as x.', 'w = 1: z دقیقاً هم‌پای x تغییر می‌کند.');
}

function Right() {
  const { tr } = useT();
  const [s, set] = useS();
  if (s.mode === 'one') {
    const z = zOf([s.w], s.b, [s.x]);
    return (
      <div className="stack">
        <Slider label="x" value={s.x} min={-4} max={4} step={0.1} color="x" onChange={(x) => set({ x })} testId="l02-x" />
        <Slider label="w" value={s.w} min={-3} max={3} step={0.05} color="w" onChange={(w) => set({ w })} testId="l02-w" />
        <Slider label="b" value={s.b} min={-5} max={5} step={0.1} color="b" onChange={(b) => set({ b })} testId="l02-b" />
        <Tex display>{`\\cz{z} = \\cw{w}\\cx{x} + \\cb{b} = \\cw{${texP(s.w, 2)}}\\cdot\\cx{${texP(s.x, 2)}} + \\cb{${texP(s.b, 2)}} = \\cz{${tex(z, 3)}}`}</Tex>
        <div className="stat-grid">
          <Stat k={tr(bi('neuron output z', 'خروجی نورون z'))} v={fmt(z, 3)} color="z" testId="l02-z" />
          <Stat k={tr(bi('slope', 'شیب'))} v={fmt(s.w, 3)} color="w" />
          <Stat k={tr(bi('z at x = 0', 'z در x = 0'))} v={fmt(s.b, 3)} color="b" />
          <Stat k={tr(bi('z = 0 at x =', 'z = 0 در x ='))} v={Math.abs(s.w) > 1e-9 ? fmt(-s.b / s.w, 3) : '—'} />
        </div>
        <Callout>{tr(slopeWords(s.w))}</Callout>
      </div>
    );
  }
  if (s.mode === 'two') {
    const t1 = s.w1 * s.x1;
    const t2 = s.w2 * s.x2;
    const z = zOf([s.w1, s.w2], s.b, [s.x1, s.x2]);
    return (
      <div className="stack">
        <Slider label="x₁" value={s.x1} min={-4} max={4} step={0.1} color="x" onChange={(x1) => set({ x1 })} />
        <Slider label="x₂" value={s.x2} min={-4} max={4} step={0.1} color="x" onChange={(x2) => set({ x2 })} />
        <Slider label="w₁" value={s.w1} min={-3} max={3} step={0.05} color="w" onChange={(w1) => set({ w1 })} />
        <Slider label="w₂" value={s.w2} min={-3} max={3} step={0.05} color="w" onChange={(w2) => set({ w2 })} />
        <Slider label="b" value={s.b} min={-5} max={5} step={0.1} color="b" onChange={(b) => set({ b })} />
        <Tex display>{`\\cz{z} = \\cw{${texP(s.w1, 2)}}\\cdot\\cx{${texP(s.x1, 2)}} + \\cw{${texP(s.w2, 2)}}\\cdot\\cx{${texP(s.x2, 2)}} + \\cb{${texP(s.b, 2)}}`}</Tex>
        <Tex display>{`= ${tex(t1, 3)} + ${texP(t2, 3)} + ${texP(s.b, 2)} = \\cz{${tex(z, 3)}}`}</Tex>
        <table className="tbl">
          <thead>
            <tr>
              <th>{tr(bi('term', 'جمله'))}</th>
              <th className="num">{tr(bi('input', 'ورودی'))}</th>
              <th className="num">{tr(bi('weight', 'وزن'))}</th>
              <th className="num">{tr(bi('contribution', 'سهم'))}</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>w₁·x₁</td>
              <td className="num c-x">{fmt(s.x1, 2)}</td>
              <td className="num c-w">{fmt(s.w1, 2)}</td>
              <td className="num">{fmtSigned(t1, 3)}</td>
            </tr>
            <tr>
              <td>w₂·x₂</td>
              <td className="num c-x">{fmt(s.x2, 2)}</td>
              <td className="num c-w">{fmt(s.w2, 2)}</td>
              <td className="num">{fmtSigned(t2, 3)}</td>
            </tr>
            <tr>
              <td>b</td>
              <td className="num">—</td>
              <td className="num">—</td>
              <td className="num c-b">{fmtSigned(s.b, 3)}</td>
            </tr>
            <tr className="hl">
              <td>z</td>
              <td />
              <td />
              <td className="num c-z">{fmt(z, 3)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    );
  }
  const f = (t: number) => zOf([s.w], s.b, [t]);
  const m = s.al * s.u + s.be * s.v;
  const lhs = f(m);
  const rhs = s.al * f(s.u) + s.be * f(s.v);
  const linear = Math.abs(s.b) < 1e-9;
  return (
    <div className="stack">
      <Slider label="w" value={s.w} min={-3} max={3} step={0.05} color="w" onChange={(w) => set({ w })} />
      <Slider label="b" value={s.b} min={-5} max={5} step={0.1} color="b" onChange={(b) => set({ b })} testId="l02-lin-b" />
      <Slider label="u" value={s.u} min={-3} max={3} step={0.1} color="x" onChange={(u) => set({ u })} />
      <Slider label="v" value={s.v} min={-3} max={3} step={0.1} color="x" onChange={(v) => set({ v })} />
      <Slider label="α" value={s.al} min={-2} max={3} step={0.1} onChange={(al) => set({ al })} />
      <Slider label="β" value={s.be} min={-2} max={3} step={0.1} onChange={(be) => set({ be })} />
      <Tex display>{`f(\\alpha u + \\beta v) = f(${tex(m, 3)}) = \\cz{${tex(lhs, 3)}}`}</Tex>
      <Tex display>{`\\alpha f(u) + \\beta f(v) = ${texP(s.al, 2)}\\cdot${texP(f(s.u), 3)} + ${texP(s.be, 2)}\\cdot${texP(f(s.v), 3)} = \\cg{${tex(rhs, 3)}}`}</Tex>
      <Tex display>{`\\text{gap} = \\cb{b}\\,(1-\\alpha-\\beta) = ${texP(s.b, 2)}\\cdot${texP(1 - s.al - s.be, 2)} = ${tex(clean(rhs - lhs), 3)}`}</Tex>
      <div className="stat-grid">
        <Stat k="f(0)" v={fmt(f(0), 3)} color="b" testId="l02-f0" />
        <Stat k={tr(bi('gap', 'فاصله'))} v={fmt(clean(rhs - lhs), 3)} testId="l02-gap" />
      </div>
      <Callout tone={linear ? 'info' : 'warn'}>
        {linear
          ? tr(bi('b = 0: the map is linear. Both sides agree for every α and β, and the line passes through the origin.', 'b = 0: نگاشت خطی است. دو طرف برای هر α و β برابرند و خط از مبدأ می‌گذرد.'))
          : tr(
              bi(
                'b ≠ 0: the map is affine, not linear. The sides only agree when α + β = 1 (try α = 0.5, β = 0.5).',
                'b ≠ 0: نگاشت آفین است، نه خطی. دو طرف فقط وقتی برابرند که α + β = 1 باشد (α = 0.5 و β = 0.5 را امتحان کن).',
              ),
            )}
      </Callout>
    </div>
  );
}

function Bottom() {
  const { tr } = useT();
  const [s, set] = useS();
  if (s.mode === 'one') {
    const z = zOf([s.w], s.b, [s.x]);
    const xs = [-2, -1, 0, 1, 2, 3];
    return (
      <div className="viz-grid">
        <div className="stack">
          <div className="small muted">{tr(bi('Where z comes from', 'z از کجا می‌آید'))}</div>
          <Waterfall terms={[{ label: '\\cw{w}\\cx{x}', v: s.w * s.x, color: 'w' }, { label: '\\cb{b}', v: s.b, color: 'b' }]} total={z} />
        </div>
        <table className="tbl">
          <thead>
            <tr>
              <th className="num c-x">x</th>
              <th className="num c-z">z</th>
              <th className="num">{tr(bi('change from previous', 'تغییر نسبت به قبلی'))}</th>
            </tr>
          </thead>
          <tbody>
            {xs.map((t, i) => (
              <tr key={t}>
                <td className="num">{t}</td>
                <td className="num">{fmt(zOf([s.w], s.b, [t]), 3)}</td>
                <td className="num c-w">{i ? fmtSigned(zOf([s.w], s.b, [t]) - zOf([s.w], s.b, [xs[i - 1]]), 3) : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  if (s.mode === 'two') {
    const z = zOf([s.w1, s.w2], s.b, [s.x1, s.x2]);
    const slice = (x2: number) => (t: number) => zOf([s.w1, s.w2], s.b, [t, x2]);
    return (
      <div className="viz-grid">
        <div className="stack">
          <div className="small muted">{tr(bi('Each input\'s contribution, added up', 'سهم هر ورودی، روی هم جمع شده'))}</div>
          <Waterfall
            terms={[
              { label: '\\cw{w_1}\\cx{x_1}', v: s.w1 * s.x1, color: 'w' },
              { label: '\\cw{w_2}\\cx{x_2}', v: s.w2 * s.x2, color: 'w' },
              { label: '\\cb{b}', v: s.b, color: 'b' },
            ]}
            total={z}
          />
        </div>
        <div>
          <div className="small muted">
            {tr(bi(`A 1-D slice: z as a function of x₁ with x₂ fixed at ${fmt(s.x2, 2)} (faint: x₂ ± 1)`, `یک برش یک‌بعدی: z بر حسب x₁ وقتی x₂ روی ${fmt(s.x2, 2)} ثابت است (کم‌رنگ: x₂ ± 1)`))}
          </div>
          <Plot2D
            xDomain={[-4, 4]}
            yDomain="auto"
            height={190}
            xLabel="x₁"
            yLabel="z"
            onDrag={(a) => set({ x1: clamp(Math.round(a * 10) / 10, -4, 4) })}
            layers={[
              { kind: 'fn', f: slice(s.x2 - 1), color: 'z', width: 1.2, opacity: 0.35 },
              { kind: 'fn', f: slice(s.x2 + 1), color: 'z', width: 1.2, opacity: 0.35 },
              { kind: 'fn', f: slice(s.x2), color: 'z', width: 2.6 },
              { kind: 'points', color: 'z', pts: [{ x: s.x1, y: z }] },
            ]}
          />
          <P className="small muted">{bi('Same slope $w_1$ for every slice; $x_2$ only shifts the line by $w_2$ per unit.', 'شیب همه برش‌ها $w_1$ است؛ $x_2$ فقط خط را به ازای هر واحد به اندازه $w_2$ جابه‌جا می‌کند.')}</P>
        </div>
      </div>
    );
  }
  const f = (t: number) => zOf([s.w], s.b, [t]);
  const tests: { name: string; al: number; be: number }[] = [
    { name: tr(bi('scaling (α = 2, β = 0)', 'مقیاس (α = 2 و β = 0)')), al: 2, be: 0 },
    { name: tr(bi('adding (α = 1, β = 1)', 'جمع (α = 1 و β = 1)')), al: 1, be: 1 },
    { name: tr(bi('average (α = β = 0.5)', 'میانگین (α = β = 0.5)')), al: 0.5, be: 0.5 },
    { name: tr(bi('your α, β', 'α و β تو')), al: s.al, be: s.be },
  ];
  return (
    <table className="tbl" data-testid="linearity-table">
      <thead>
        <tr>
          <th>{tr(bi('test', 'آزمون'))}</th>
          <th className="num">f(αu+βv)</th>
          <th className="num">αf(u)+βf(v)</th>
          <th className="num">{tr(bi('gap', 'فاصله'))}</th>
          <th>{tr(bi('equal?', 'برابر؟'))}</th>
        </tr>
      </thead>
      <tbody>
        {tests.map((t) => {
          const l = f(t.al * s.u + t.be * s.v);
          const r = t.al * f(s.u) + t.be * f(s.v);
          return (
            <tr key={t.name}>
              <td>{t.name}</td>
              <td className="num c-z">{fmt(l, 3)}</td>
              <td className="num c-g">{fmt(r, 3)}</td>
              <td className="num c-b">{fmt(clean(r - l), 3)}</td>
              <td>{Math.abs(r - l) < 1e-9 ? '✓' : '✗'}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export const lesson02: LessonDef = {
  id: ID,
  title: bi('Weight and Bias', 'وزن و بایاس'),
  short: bi('Weight & bias', 'وزن و بایاس'),
  objectives: [
    bi('Compute z = w·x + b and explain what w and b each control', 'z = w·x + b را حساب کنی و توضیح بدهی w و b هر کدام چه چیزی را کنترل می‌کنند'),
    bi('Predict the effect of a negative weight and of a larger bias', 'اثر وزن منفی و بایاس بزرگ‌تر را پیش‌بینی کنی'),
    bi('Add up the contributions of several inputs: z = w₁x₁ + w₂x₂ + b', 'سهم چند ورودی را جمع کنی: z = w₁x₁ + w₂x₂ + b'),
    bi('Tell a linear map from an affine one, precisely', 'نگاشت خطی را دقیقاً از نگاشت آفین تشخیص بدهی'),
  ],
  revealSteps: 6,
  Explain,
  Center,
  Right,
  Bottom,
  checkpoint: {
    question: bi('Which of these is linear in the strict mathematical sense?', 'کدام یک در معنای دقیق ریاضی خطی است؟'),
    choices: [
      { id: 'a', label: bi('$z = 3x + 2$', '$z = 3x + 2$') },
      { id: 'b', label: bi('$z = -2x$', '$z = -2x$') },
      { id: 'c', label: bi('$z = x + 0.1$', '$z = x + 0.1$') },
    ],
    answer: 'b',
    explain: bi(
      'Only $z = -2x$ has $f(0) = 0$ and satisfies $f(\\alpha u + \\beta v) = \\alpha f(u) + \\beta f(v)$ for all $\\alpha, \\beta$. The others are affine: a nonzero bias breaks the rule by $b(1 - \\alpha - \\beta)$.',
      'فقط $z = -2x$ است که $f(0) = 0$ دارد و برای همه $\\alpha$ و $\\beta$ رابطه $f(\\alpha u + \\beta v) = \\alpha f(u) + \\beta f(v)$ را برقرار می‌کند. بقیه آفین‌اند: بایاس غیرصفر قاعده را به اندازه $b(1 - \\alpha - \\beta)$ به هم می‌زند.',
    ),
  },
};
