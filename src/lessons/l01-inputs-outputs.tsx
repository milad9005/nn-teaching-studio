// Lesson 01: Inputs and outputs. A function as a rule (a pure function in code), coordinates on
// two axes, and a graph as the picture of every input-output pair. Lesson-local state only.

import { useMemo } from 'react';
import type { LessonDef } from './types';
import { bi, useT, type Bi } from '../i18n';
import { useLessonState } from '../state/lessonState';
import { Plot2D, type PlotLayer } from '../components/Plot2D';
import { Tex } from '../components/Tex';
import { Segmented, Slider, Stat } from '../components/controls';
import { Advanced, Callout, Cycle, P, Predict, Reveal } from '../components/teaching';
import { clamp, fmt, fmtSigned, linspace, tex, texP } from '../engine';

const ID = 'l01';

type FnId = 'double' | 'plus3' | 'square' | 'fan';

interface FnDef {
  f: (x: number) => number;
  /** The rule in TeX. */
  formula: string;
  /** The rule with the current input substituted, ending in the output. */
  subst: (x: number) => string;
  code: string;
  call: (x: number) => string;
  xd: [number, number];
  yd: [number, number];
  step: number;
  x0: number;
  /** Inputs listed in the value table. */
  table: number[];
  xName: Bi;
  yName: Bi;
  inVar: string;
  outVar: string;
}

const fanSpeed = (T: number) => clamp(10 * (T - 20), 0, 100);

const FNS: Record<FnId, FnDef> = {
  double: {
    f: (x) => 2 * x,
    formula: '\\cyhat{y} = 2\\,\\cx{x}',
    subst: (x) => `\\cyhat{y} = 2\\cdot\\cx{${texP(x, 2)}} = \\cyhat{${tex(2 * x, 2)}}`,
    code: 'const f = (x) => 2 * x;',
    call: (x) => `f(${fmt(x, 2).replace('−', '-')})  // → ${fmt(2 * x, 2).replace('−', '-')}`,
    xd: [-5, 5],
    yd: [-10, 10],
    step: 0.1,
    x0: 3,
    table: [-3, -2, -1, 0, 1, 2, 3],
    xName: bi('input x', 'ورودی x'),
    yName: bi('output y', 'خروجی y'),
    inVar: 'x',
    outVar: 'y',
  },
  plus3: {
    f: (x) => x + 3,
    formula: '\\cyhat{y} = \\cx{x} + 3',
    subst: (x) => `\\cyhat{y} = \\cx{${tex(x, 2)}} + 3 = \\cyhat{${tex(x + 3, 2)}}`,
    code: 'const f = (x) => x + 3;',
    call: (x) => `f(${fmt(x, 2).replace('−', '-')})  // → ${fmt(x + 3, 2).replace('−', '-')}`,
    xd: [-5, 5],
    yd: [-3, 9],
    step: 0.1,
    x0: 1,
    table: [-3, -2, -1, 0, 1, 2, 3],
    xName: bi('input x', 'ورودی x'),
    yName: bi('output y', 'خروجی y'),
    inVar: 'x',
    outVar: 'y',
  },
  square: {
    f: (x) => x * x,
    formula: '\\cyhat{y} = \\cx{x}^2',
    subst: (x) => `\\cyhat{y} = \\cx{${texP(x, 2)}}^2 = \\cyhat{${tex(x * x, 2)}}`,
    code: 'const f = (x) => x * x;',
    call: (x) => `f(${fmt(x, 2).replace('−', '-')})  // → ${fmt(x * x, 2).replace('−', '-')}`,
    xd: [-4, 4],
    yd: [-1, 16],
    step: 0.1,
    x0: -2,
    table: [-3, -2, -1, 0, 1, 2, 3],
    xName: bi('input x', 'ورودی x'),
    yName: bi('output y', 'خروجی y'),
    inVar: 'x',
    outVar: 'y',
  },
  fan: {
    f: fanSpeed,
    formula: 's(\\cx{T}) = \\begin{cases} 0 & T \\le 20 \\\\ 10\\,(T-20) & 20 < T < 30 \\\\ 100 & T \\ge 30 \\end{cases}',
    subst: (T) => {
      const s = tex(fanSpeed(T), 2);
      if (T <= 20) return `s(\\cx{${tex(T, 2)}}) = \\cyhat{0}\\quad(${tex(T, 2)} \\le 20)`;
      if (T >= 30) return `s(\\cx{${tex(T, 2)}}) = \\cyhat{100}\\quad(${tex(T, 2)} \\ge 30)`;
      return `s(\\cx{${tex(T, 2)}}) = 10\\cdot(${tex(T, 2)} - 20) = \\cyhat{${s}}`;
    },
    code: 'const fan = (T) => Math.min(100, Math.max(0, 10 * (T - 20)));',
    call: (T) => `fan(${fmt(T, 2)})  // → ${fmt(fanSpeed(T), 2)}`,
    xd: [10, 35],
    yd: [-10, 110],
    step: 0.5,
    x0: 24,
    table: [14, 18, 20, 22, 26, 30, 34],
    xName: bi('temperature T (°C)', 'دما T (درجه سانتی‌گراد)'),
    yName: bi('fan speed s (%)', 'سرعت پنکه s (درصد)'),
    inVar: 'T',
    outVar: 's',
  },
};

interface S {
  fn: FnId;
  x: number;
  /** Inputs the learner pinned; outputs are always recomputed from the current rule. */
  pins: number[];
  curve: boolean;
}
const INIT: S = { fn: 'double', x: 3, pins: [], curve: true };
const useS = () => useLessonState<S>(ID, INIT);

const snap = (v: number, step: number) => Math.round(v / step) * step;

function Explain() {
  const { tr } = useT();
  const [s, set] = useS();
  return (
    <>
      <Cycle active={0} />
      <Reveal at={1}>
        <h3>{tr(bi('1. A function is a rule', '۱. تابع یعنی یک قاعده'))}</h3>
        <P>
          {bi(
            'A function takes a value in, follows a fixed rule, and gives a value out. The value that goes in is the input; the value that comes out is the output. "Double it" is a function: put in 3, get out 6.',
            'تابع (function) یک مقدار را می‌گیرد، طبق یک قاعده ثابت با آن کار می‌کند و یک مقدار پس می‌دهد. چیزی که وارد می‌شود ورودی (input) است و چیزی که بیرون می‌آید خروجی (output). «دو برابرش کن» یک تابع است: ۳ بده، ۶ بگیر.',
          )}
        </P>
        <P>
          {bi(
            'You already know this from code. A pure function like $\\texttt{(x) => 2 * x}$ always returns the same output for the same input. In math we write the same rule as $\\cyhat{y} = 2\\cx{x}$.',
            'این را از برنامه‌نویسی می‌شناسی. یک تابع خالص (pure function) مثل $\\texttt{(x) => 2 * x}$ برای یک ورودی همیشه همان خروجی را برمی‌گرداند. در ریاضی همین قاعده را این‌طور می‌نویسیم: $\\cyhat{y} = 2\\cx{x}$.',
          )}
        </P>
        <P>
          {bi(
            'The letters are variables: names for values that can change, just like parameters in code. $\\cx{x}$ names the input, $\\cyhat{y}$ names the output. Drag $x$ on the right and watch $y$ follow.',
            'حرف‌ها متغیر (variable) هستند: اسم‌هایی برای مقدارهایی که می‌توانند عوض شوند، درست مثل پارامترهای یک تابع در کد. $\\cx{x}$ اسم ورودی است و $\\cyhat{y}$ اسم خروجی. $x$ را در سمت راست جابه‌جا کن و ببین $y$ چطور دنبالش می‌آید.',
          )}
        </P>
      </Reveal>
      <Reveal at={2}>
        <h3>{tr(bi('2. Two axes, one point', '۲. دو محور، یک نقطه'))}</h3>
        <P>
          {bi(
            'To draw a function we use two number lines. The horizontal one (the x-axis) holds the input. The vertical one (the y-axis) holds the output. They cross at $(0, 0)$, the origin.',
            'برای کشیدن یک تابع از دو محور عددی استفاده می‌کنیم. محور افقی (محور x) جای ورودی است و محور عمودی (محور y) جای خروجی. این دو در نقطه $(0, 0)$ به هم می‌رسند که به آن مبدأ (origin) می‌گوییم.',
          )}
        </P>
        <P>
          {bi(
            'A point is written $(\\cx{x}, \\cyhat{y})$: input first, output second. $(3, 6)$ means "go 3 to the right, then 6 up". Negative numbers go left or down. The dashed lines on the plot show exactly this walk.',
            'هر نقطه را به شکل $(\\cx{x}, \\cyhat{y})$ می‌نویسیم: اول ورودی، بعد خروجی. $(3, 6)$ یعنی «۳ واحد به راست برو، بعد ۶ واحد بالا». عددهای منفی یعنی به چپ یا پایین. خط‌چین‌های روی نمودار دقیقاً همین مسیر را نشان می‌دهند.',
          )}
        </P>
      </Reveal>
      <Reveal at={3}>
        <h3>{tr(bi('3. A graph is every pair at once', '۳. نمودار یعنی همه جفت‌ها با هم'))}</h3>
        <P>
          {bi(
            'Turn off "show the curve" and press "add point" for several inputs. Each press drops one (input, output) pair. With enough of them the dots line up into the curve: a graph is simply the picture of all input-output pairs.',
            '«نمایش منحنی» را خاموش کن و برای چند ورودی مختلف «افزودن نقطه» را بزن. هر بار یک جفت (ورودی، خروجی) روی نمودار می‌نشیند. با تعداد کافی، نقطه‌ها کنار هم شکل منحنی را می‌سازند: نمودار فقط تصویر همه جفت‌های ورودی و خروجی است.',
          )}
        </P>
        <Predict
          id={ID + '-square'}
          question={bi(
            'In $y = x^2$ the input increases from $x = -2$ to $x = -1$. Does the output go up or down?',
            'در $y = x^2$ ورودی از $x = -2$ به $x = -1$ زیاد می‌شود. خروجی بالا می‌رود یا پایین؟',
          )}
          choices={[
            { id: 'up', label: bi('Up', 'بالا') },
            { id: 'down', label: bi('Down', 'پایین') },
            { id: 'same', label: bi('Stays the same', 'تغییری نمی‌کند') },
          ]}
          answer="down"
          explain={bi(
            '$(-2)^2 = 4$ but $(-1)^2 = 1$, so the output drops even though the input grew. Pick $y = x^2$ and drag $x$ from −2 to −1 to see it. A bigger input does not always mean a bigger output: it depends on the rule.',
            '$(-2)^2 = 4$ ولی $(-1)^2 = 1$، پس خروجی پایین می‌آید با اینکه ورودی بزرگ‌تر شده. $y = x^2$ را انتخاب کن و $x$ را از ۲− تا ۱− بکش. ورودی بزرگ‌تر همیشه خروجی بزرگ‌تر نمی‌دهد: به قاعده بستگی دارد.',
          )}
        />
      </Reveal>
      <Reveal at={4}>
        <h3>{tr(bi('4. A real rule: temperature → fan speed', '۴. یک قاعده واقعی: دما ← سرعت پنکه'))}</h3>
        <P>
          {bi(
            'A smart fan reads the temperature $\\cx{T}$ and sets its speed $s$: off up to 20 °C, then 10 % faster per degree, and full speed from 30 °C. Input and output now have units, and the rule is made of three simple pieces.',
            'یک پنکه هوشمند دما $\\cx{T}$ را می‌خواند و سرعتش $s$ را تنظیم می‌کند: تا ۲۰ درجه خاموش است، بعد به ازای هر درجه ۱۰ درصد تندتر، و از ۳۰ درجه به بعد با حداکثر سرعت. حالا ورودی و خروجی واحد دارند و قاعده از سه تکه ساده ساخته شده است.',
          )}
        </P>
        {s.fn !== 'fan' && (
          <button className="btn sm" onClick={() => set({ fn: 'fan', x: FNS.fan.x0, pins: [] })} data-testid="to-fan">
            {tr(bi('Try the fan rule →', 'قاعده پنکه را امتحان کن ←'))}
          </button>
        )}
        <Callout>
          {tr(
            bi(
              'A neural network is also a function: numbers in (like a temperature), numbers out (like a prediction). The difference is that nobody writes its rule by hand. It has adjustable knobs, and it learns their values from examples. Shapes like "flat, then rising" are exactly what its building blocks produce (lessons 3 and 4).',
              'شبکه عصبی هم یک تابع است: عدد می‌گیرد (مثلاً دما) و عدد می‌دهد (مثلاً یک پیش‌بینی). فرقش این است که کسی قاعده‌اش را با دست نمی‌نویسد. پیچ‌های قابل تنظیم دارد و مقدار آنها را از روی مثال‌ها یاد می‌گیرد. شکل‌هایی مثل «اول صاف، بعد رو به بالا» دقیقاً همان چیزی است که اجزای سازنده شبکه می‌سازند (درس ۳ و ۴).',
            ),
          )}
        </Callout>
      </Reveal>
      <Reveal at={5}>
        <Advanced>
          <P>
            {bi(
              'Formally, a function $f: D \\to R$ assigns to every input in its domain $D$ exactly one output in $R$. We write $y = f(x)$. Here $D = \\mathbb{R}$ for the formulas and $D = [10, 35]$ for the fan as plotted.',
              'به زبان دقیق، تابع $f: D \\to R$ به هر ورودی در دامنه (domain) $D$ دقیقاً یک خروجی در $R$ نسبت می‌دهد. می‌نویسیم $y = f(x)$. اینجا برای فرمول‌ها $D = \\mathbb{R}$ است و برای پنکه، در این نمودار، $D = [10, 35]$.',
            )}
          </P>
          <P>
            {bi(
              '"Exactly one output" is why a vertical line crosses a function\'s graph at most once. Different inputs may share an output: $f(-2) = f(2) = 4$ for $f(x) = x^2$ is fine.',
              '«دقیقاً یک خروجی» یعنی هر خط عمودی نمودار تابع را حداکثر در یک نقطه قطع می‌کند. ولی دو ورودی متفاوت می‌توانند خروجی یکسان داشته باشند: برای $f(x) = x^2$ داریم $f(-2) = f(2) = 4$ و این مشکلی ندارد.',
            )}
          </P>
          <P>
            {bi(
              'Functions can take several inputs, $f(x_1, x_2)$, or a whole vector $f(\\mathbf{x})$, and return several outputs. A network with parameters $\\theta$ is written $\\hat y = f_\\theta(\\mathbf{x})$.',
              'تابع‌ها می‌توانند چند ورودی بگیرند، مثل $f(x_1, x_2)$، یا یک بردار کامل $f(\\mathbf{x})$، و چند خروجی بدهند. یک شبکه با پارامترهای $\\theta$ را به شکل $\\hat y = f_\\theta(\\mathbf{x})$ می‌نویسیم.',
            )}
          </P>
        </Advanced>
      </Reveal>
    </>
  );
}

function Center() {
  const { tr } = useT();
  const [s, set] = useS();
  const d = FNS[s.fn];
  const y = d.f(s.x);
  const layers = useMemo<PlotLayer[]>(() => {
    const out: PlotLayer[] = [];
    if (s.curve) out.push({ kind: 'fn', f: d.f, color: 'yhat', width: 2.6 });
    // The walk to the point: along the input axis, then up (or down) to the output.
    const baseY = d.yd[0] < 0 && d.yd[1] > 0 ? 0 : d.yd[0];
    const baseX = d.xd[0] < 0 && d.xd[1] > 0 ? 0 : d.xd[0];
    out.push(
      { kind: 'segment', from: [s.x, baseY], to: [s.x, y], color: 'x', dash: '5 4', width: 1.6 },
      { kind: 'segment', from: [baseX, y], to: [s.x, y], color: 'yhat', dash: '5 4', width: 1.6 },
      { kind: 'points', color: 'x', pts: [{ x: s.x, y: baseY, r: 4, label: `${d.inVar} = ${fmt(s.x, 2)}` }] },
      { kind: 'points', color: 'yhat', pts: s.pins.map((px) => ({ x: px, y: d.f(px), r: 4.5 })) },
      { kind: 'text', x: s.x, y, text: `(${fmt(s.x, 2)}, ${fmt(y, 2)})`, color: 'var(--ink)', dy: -14, anchor: 'middle' },
    );
    return out;
  }, [s, d, y]);
  return (
    <div className="viz">
      <div className="row">
        <Segmented
          label="function"
          value={s.fn}
          onChange={(fn) => set({ fn, x: FNS[fn].x0, pins: [] })}
          options={[
            { value: 'double', label: <Tex>{'y = 2x'}</Tex> },
            { value: 'plus3', label: <Tex>{'y = x + 3'}</Tex> },
            { value: 'square', label: <Tex>{'y = x^2'}</Tex> },
            { value: 'fan', label: tr(bi('Temperature → fan', 'دما ← پنکه')) },
          ]}
        />
      </div>
      <div className="formula-hero" data-testid="machine">
        <Tex>{`\\cx{${tex(s.x, 2)}} \\;\\longrightarrow\\; \\boxed{\\;${s.fn === 'fan' ? 's(\\cx{T})' : d.formula}\\;} \\;\\longrightarrow\\; \\cyhat{${tex(y, 2)}}`}</Tex>
      </div>
      <Plot2D
        testId="io-plot"
        xDomain={d.xd}
        yDomain={d.yd}
        layers={layers}
        height={340}
        xLabel={tr(d.xName)}
        yLabel={tr(d.yName)}
        handle={{ x: s.x, y, color: 'yhat' }}
        onDrag={(x) => set({ x: clamp(snap(x, d.step), d.xd[0], d.xd[1]) })}
        ariaLabel="function plot"
      />
      <div className="row">
        <button className="btn sm primary" onClick={() => set({ pins: s.pins.includes(s.x) ? s.pins : [...s.pins, s.x] })} data-testid="add-pin">
          + {tr(bi('Add this point', 'افزودن این نقطه'))}
        </button>
        <button className="btn sm ghost" onClick={() => set({ pins: [] })} disabled={!s.pins.length}>
          {tr(bi('Clear points', 'پاک کردن نقطه‌ها'))}
        </button>
        <label className="small row" style={{ gap: 4 }}>
          <input type="checkbox" checked={s.curve} onChange={(e) => set({ curve: e.target.checked })} data-testid="show-curve" />
          {tr(bi('Show the curve', 'نمایش منحنی'))}
        </label>
      </div>
      <p className="small muted">{tr(bi('Drag on the plot to change the input.', 'روی نمودار بکش تا ورودی عوض شود.'))}</p>
    </div>
  );
}

function Right() {
  const { tr } = useT();
  const [s, set] = useS();
  const d = FNS[s.fn];
  const y = d.f(s.x);
  const dy = d.f(s.x + 1) - y;
  return (
    <div className="stack">
      <Slider
        label={tr(d.xName)}
        value={s.x}
        min={d.xd[0]}
        max={d.xd[1]}
        step={d.step}
        color="x"
        digits={2}
        onChange={(x) => set({ x })}
        testId="slider-x"
      />
      <h4>{tr(bi('The rule', 'قاعده'))}</h4>
      <Tex display>{d.formula}</Tex>
      <h4>{tr(bi('Plug in the input', 'ورودی را جایگذاری کن'))}</h4>
      <Tex display>{d.subst(s.x)}</Tex>
      <h4>{tr(bi('The same rule in code', 'همین قاعده در کد'))}</h4>
      <pre className="mono small card flat" dir="ltr" style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
        {d.code}
        {'\n'}
        {d.call(s.x)}
      </pre>
      <div className="stat-grid">
        <Stat k={tr(bi('input', 'ورودی'))} v={fmt(s.x, 2)} color="x" testId="io-input" />
        <Stat k={tr(bi('output', 'خروجی'))} v={fmt(y, 2)} color="yhat" testId="io-output" />
        <Stat k={tr(bi('output change for input +1', 'تغییر خروجی با ۱+ ورودی'))} v={fmtSigned(dy, 2)} testId="io-step" />
      </div>
      <P className="small muted">
        {s.fn === 'square'
          ? bi(
              'For $y = x^2$ the change for "+1" depends on where you are: try $x = -2$, $0$ and $2$.',
              'برای $y = x^2$ تغییر به ازای «۱+» به جایی که هستی بستگی دارد: $x = -2$، $0$ و $2$ را امتحان کن.',
            )
          : s.fn === 'fan'
            ? bi(
                'In the middle piece every extra degree adds exactly 10 %. Below 20 °C and above 30 °C, extra degrees change nothing.',
                'در تکه وسط هر درجه اضافه دقیقاً ۱۰ درصد اضافه می‌کند. زیر ۲۰ و بالای ۳۰ درجه، درجه اضافه هیچ تغییری نمی‌دهد.',
              )
            : bi(
                'For this rule "+1" on the input always changes the output by the same amount, wherever you are. That is why its graph is a straight line.',
                'در این قاعده «۱+» روی ورودی، هر جا که باشی، خروجی را همیشه به یک اندازه تغییر می‌دهد. برای همین نمودارش یک خط راست است.',
              )}
      </P>
    </div>
  );
}

function Bottom() {
  const { tr } = useT();
  const [s, set] = useS();
  const d = FNS[s.fn];
  const pins = [...s.pins].sort((a, b) => a - b);
  return (
    <div className="viz-grid">
      <div>
        <div className="small muted">{tr(bi('A table of the same function (click a row to jump there)', 'جدول همین تابع (روی هر سطر بزن تا به آنجا بروی)'))}</div>
        <table className="tbl" data-testid="value-table">
          <thead>
            <tr>
              <th className="num c-x">{d.inVar}</th>
              <th className="num c-yhat">{d.outVar}</th>
              <th className="num">{tr(bi('change from previous row', 'تغییر نسبت به سطر قبل'))}</th>
            </tr>
          </thead>
          <tbody>
            {d.table.map((v, i) => (
              <tr key={v} className={Math.abs(v - s.x) < 1e-9 ? 'hl' : ''} onClick={() => set({ x: v })} style={{ cursor: 'pointer' }}>
                <td className="num">{fmt(v, 2)}</td>
                <td className="num">{fmt(d.f(v), 2)}</td>
                <td className="num muted">{i === 0 ? '' : fmtSigned(d.f(v) - d.f(d.table[i - 1]), 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div>
        <div className="small muted">{tr(bi('Your pinned points', 'نقطه‌هایی که اضافه کرده‌ای'))}</div>
        {pins.length ? (
          <table className="tbl" data-testid="pin-table">
            <thead>
              <tr>
                <th>{tr(bi('point', 'نقطه'))}</th>
                <th className="num c-x">{d.inVar}</th>
                <th className="num c-yhat">{d.outVar}</th>
              </tr>
            </thead>
            <tbody>
              {pins.map((v) => (
                <tr key={v}>
                  <td className="mono" dir="ltr">
                    ({fmt(v, 2)}, {fmt(d.f(v), 2)})
                  </td>
                  <td className="num">{fmt(v, 2)}</td>
                  <td className="num">{fmt(d.f(v), 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <P className="muted small">{bi('No points yet. Press "Add this point" under the plot.', 'هنوز نقطه‌ای نیست. دکمه «افزودن این نقطه» زیر نمودار را بزن.')}</P>
        )}
        <Plot2D
          xDomain={d.xd}
          yDomain={d.yd}
          height={150}
          layers={[
            { kind: 'points', color: 'yhat', pts: linspace(d.xd[0], d.xd[1], 11).map((v) => ({ x: v, y: d.f(v), r: 3 })) },
            { kind: 'points', color: 'x', pts: pins.map((v) => ({ x: v, y: d.f(v), r: 4 })) },
          ]}
          xLabel={d.inVar}
          yLabel={d.outVar}
          ariaLabel="sampled points"
        />
        <P className="small muted">{bi('Only 11 sampled pairs, and the shape is already visible.', 'فقط ۱۱ جفت نمونه، و شکل از همین حالا پیداست.')}</P>
      </div>
    </div>
  );
}

export const lesson01: LessonDef = {
  id: ID,
  title: bi('Inputs and Outputs', 'ورودی و خروجی'),
  short: bi('Inputs & outputs', 'ورودی و خروجی'),
  objectives: [
    bi('Describe a function as a rule that turns an input into exactly one output', 'تابع را به شکل قاعده‌ای توضیح بدهی که هر ورودی را به دقیقاً یک خروجی تبدیل می‌کند'),
    bi('Read and place points (x, y) on two axes', 'نقطه‌های (x, y) را روی دو محور بخوانی و پیدا کنی'),
    bi('See a graph as the picture of all input-output pairs', 'نمودار را تصویر همه جفت‌های ورودی و خروجی ببینی'),
    bi('Predict how the output changes when the input changes', 'پیش‌بینی کنی با تغییر ورودی، خروجی چطور تغییر می‌کند'),
  ],
  revealSteps: 5,
  Explain,
  Center,
  Right,
  Bottom,
  checkpoint: {
    question: bi('The point $(2, -1)$ lies on the graph of a function $f$. What does that tell you?', 'نقطه $(2, -1)$ روی نمودار تابع $f$ است. این به ما چه می‌گوید؟'),
    choices: [
      { id: 'a', label: bi('$f(2) = -1$: input 2 gives output −1', '$f(2) = -1$: ورودی ۲ خروجی ۱− می‌دهد') },
      { id: 'b', label: bi('$f(-1) = 2$: input −1 gives output 2', '$f(-1) = 2$: ورودی ۱− خروجی ۲ می‌دهد') },
      { id: 'c', label: bi('2 and −1 are both inputs', '۲ و ۱− هر دو ورودی‌اند') },
    ],
    answer: 'a',
    explain: bi(
      'Points are written (input, output): go 2 to the right along the x-axis, then 1 down to the output.',
      'نقطه‌ها به شکل (ورودی، خروجی) نوشته می‌شوند: ۲ واحد روی محور x به راست برو، بعد ۱ واحد پایین تا به خروجی برسی.',
    ),
  },
};
