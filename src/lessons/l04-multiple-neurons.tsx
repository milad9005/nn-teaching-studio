// Lesson 04: Multiple neurons. One input, a hidden ReLU layer of 1-10 neurons, one linear output.
// Each neuron adds a hinge v_k ReLU(w_k x + b_k); the network output (engine predict) is their sum.
// Starts from preset 'one-vs-many' (target y = 0.5 x^2). Lesson-local state.

import { useMemo } from 'react';
import type { LessonDef } from './types';
import { bi, useT } from '../i18n';
import { useLessonState } from '../state/lessonState';
import { Plot2D, type PlotLayer } from '../components/Plot2D';
import { Tex } from '../components/Tex';
import { Slider, Stat } from '../components/controls';
import { Advanced, Callout, Cycle, P, Predict, Reveal } from '../components/teaching';
import { act, clamp, datasetLoss, fmt, fmtSigned, loadPreset, predict, tex, texP } from '../engine';
import { activeSide, breakpoint, buildNet, contribution, kinks, MAX_UNITS, scenario, type ScenarioId, type Unit } from './math/l04';

const ID = 'l04';
const XD: [number, number] = [-3, 3];
const DATA = loadPreset('one-vs-many').data;

interface S {
  n: number;
  units: Unit[];
  c: number;
  sel: number;
  probe: number;
  parts: boolean;
}
const start = scenario('preset');
const INIT: S = { n: start.n, units: start.units, c: start.c, sel: 0, probe: 1, parts: true };
const useS = () => useLessonState<S>(ID, INIT);

const pieceColor = (u: Unit) => (u.v > 1e-12 ? 'wpos' : u.v < -1e-12 ? 'wneg' : 'var(--muted)');

/** Everything the panels show, computed once from the lesson state with the engine network. */
function useModel(s: S) {
  return useMemo(() => {
    const used = s.units.slice(0, s.n);
    const net = buildNet(used, s.c);
    const f = (x: number) => predict(net, [x])[0];
    const bends = kinks(used, XD[0], XD[1]);
    return { used, net, f, bends, mse: datasetLoss(net, DATA, 'mse') };
  }, [s.units, s.n, s.c]);
}

function Explain() {
  const { tr } = useT();
  return (
    <>
      <Cycle active={2} />
      <Reveal at={1}>
        <h3>{tr(bi('1. Many hinges, one output', '۱. چند لولا، یک خروجی'))}</h3>
        <P>
          {bi(
            'Put several ReLU neurons side by side. They all read the same input $\\cx{x}$, but each has its own $\\cw{w_k}$ and $\\cb{b_k}$, so each one is a hinge: flat on one side of its breakpoint $x = -b_k/w_k$, a straight ramp on the other.',
            'چند نورون ReLU را کنار هم بگذار. همه یک ورودی $\\cx{x}$ را می‌خوانند ولی هر کدام $\\cw{w_k}$ و $\\cb{b_k}$ خودش را دارد، پس هر کدام یک لولا (hinge) است: یک طرف نقطه شکستش $x = -b_k/w_k$ صاف است و طرف دیگر یک سطح شیب‌دار.',
          )}
        </P>
        <P>
          {bi(
            'A linear output neuron then adds the hinges up, each scaled by its output weight $\\cw{v_k}$:',
            'بعد یک نورون خروجی خطی این لولاها را با هم جمع می‌کند و هر کدام را در وزن خروجی خودش $\\cw{v_k}$ ضرب می‌کند:',
          )}
        </P>
        <Tex display>{'\\cyhat{\\hat y}(x) = \\cb{c} + \\sum_{k=1}^{n} \\cw{v_k}\\,\\relu(\\cw{w_k}\\cx{x} + \\cb{b_k})'}</Tex>
      </Reveal>
      <Reveal at={2}>
        <h3>{tr(bi('2. What each knob does', '۲. هر پیچ چه می‌کند'))}</h3>
        <P>
          {bi(
            'The sign of $w_k$ picks the side where the neuron is on ($w_k > 0$: right of the breakpoint, $w_k < 0$: left). $b_k$ slides the breakpoint. $v_k$ scales the piece, and a negative $v_k$ turns it upside down. Teal pieces have $v_k > 0$, orange ones $v_k < 0$.',
            'علامت $w_k$ تعیین می‌کند نورون در کدام طرف روشن است ($w_k > 0$: سمت راست نقطه شکست، $w_k < 0$: سمت چپ). $b_k$ نقطه شکست را جابه‌جا می‌کند. $v_k$ اندازه تکه را تعیین می‌کند و $v_k$ منفی آن را وارونه می‌کند. تکه‌های سبزآبی $v_k > 0$ دارند و نارنجی‌ها $v_k < 0$.',
          )}
        </P>
        <Predict
          id={ID + '-flip'}
          question={bi('Select neuron 3 and change its output weight $v_3$ from $+0.5$ to $-0.5$. What happens to its piece?', 'نورون ۳ را انتخاب کن و وزن خروجی‌اش $v_3$ را از $+0.5$ به $-0.5$ ببر. تکه‌اش چه می‌شود؟')}
          choices={[
            { id: 'flip', label: bi('It flips upside down: the ramp now goes the other way', 'وارونه می‌شود: سطح شیب‌دار حالا به سمت دیگر می‌رود') },
            { id: 'move', label: bi('Its breakpoint moves', 'نقطه شکستش جابه‌جا می‌شود') },
            { id: 'off', label: bi('The neuron switches off', 'نورون خاموش می‌شود') },
          ]}
          answer="flip"
          explain={bi(
            'The neuron itself (where it is on, where it bends) depends only on $w_3$ and $b_3$. $v_3$ only multiplies its output, so the piece is mirrored across the x-axis and the total curve now bends down at that breakpoint instead of up.',
            'خود نورون (اینکه کجا روشن است و کجا می‌شکند) فقط به $w_3$ و $b_3$ بستگی دارد. $v_3$ فقط خروجی‌اش را ضرب می‌کند، پس تکه نسبت به محور x قرینه می‌شود و منحنی کل در آن نقطه به جای بالا به پایین خم می‌شود.',
          )}
        />
      </Reveal>
      <Reveal at={3}>
        <h3>{tr(bi('3. Bends add up to a curve', '۳. خم‌ها روی هم یک منحنی می‌سازند'))}</h3>
        <P>
          {bi(
            'The sum is piecewise linear: straight between breakpoints, bending only at them. At neuron $k$\'s breakpoint the slope changes by exactly $v_k\\,|w_k|$. With enough well-placed bends, the straight pieces can follow a curve.',
            'حاصل‌جمع تکه‌ای‌خطی (piecewise linear) است: بین نقطه‌های شکست صاف است و فقط روی آنها خم می‌شود. در نقطه شکست نورون $k$ شیب دقیقاً به اندازه $v_k\\,|w_k|$ تغییر می‌کند. با تعداد کافی خم در جاهای درست، تکه‌های صاف می‌توانند یک منحنی را دنبال کنند.',
          )}
        </P>
        <Callout>
          {tr(
            bi(
              'Challenge: fit the gold points of y = 0.5x². The curve gets steeper as |x| grows, so the outer hinges need larger v. Try it by hand, then compare with "Best with 4" and "Best with 8".',
              'چالش: نقطه‌های طلایی y = 0.5x² را بپوشان. منحنی با بزرگ شدن |x| تندتر می‌شود، پس لولاهای بیرونی v بزرگ‌تری لازم دارند. اول خودت امتحان کن، بعد با «بهترین با ۴» و «بهترین با ۸» مقایسه کن.',
            ),
          )}
        </Callout>
      </Reveal>
      <Reveal at={4}>
        <h3>{tr(bi('4. More neurons is not automatically better', '۴. نورون بیشتر خودبه‌خود بهتر نیست'))}</h3>
        <P>
          {bi(
            'More neurons raise the capacity: the most bends the network could make. What it actually computes depends on the weights. Try the three scenarios: duplicated neurons, a breakpoint outside the range we look at, and output weights of 0. Each uses 6 or 8 neurons, yet draws exactly the same function as "Best with 4": same bends, same error.',
            'نورون بیشتر ظرفیت (capacity) را بالا می‌برد: یعنی بیشترین تعداد خمی که شبکه می‌تواند بسازد. اما چیزی که واقعاً حساب می‌کند به وزن‌ها بستگی دارد. سه سناریو را امتحان کن: نورون‌های تکراری، نقطه شکستی بیرون از بازه‌ای که نگاه می‌کنیم، و وزن خروجی ۰. هر کدام ۶ یا ۸ نورون دارند ولی دقیقاً همان تابع «بهترین با ۴» را می‌کشند: همان خم‌ها، همان خطا.',
          )}
        </P>
        <P>
          {bi(
            'And accuracy is a separate question: extra neurons only help if training places them well. Passing exactly through every training point is not even the real goal; we come back to that with overfitting.',
            'دقت هم سؤال جداگانه‌ای است: نورون‌های اضافه فقط وقتی کمک می‌کنند که آموزش آنها را در جای درست بگذارد. حتی گذشتن دقیق از همه نقطه‌های آموزشی هم هدف واقعی نیست؛ در بحث بیش‌برازش (overfitting) به این برمی‌گردیم.',
          )}
        </P>
      </Reveal>
      <Reveal at={5}>
        <Advanced>
          <P>
            {bi(
              'With one input, $n$ hidden ReLU neurons give at most $n$ breakpoints, hence at most $n + 1$ linear pieces. Neurons with $w_k = 0$ are constant ($\\relu(b_k)$) and only shift the output. Two neurons can cancel: $\\relu(x) - \\relu(-x) = x$ has no bend, while $\\relu(x) + \\relu(-x) = |x|$ has one.',
              'با یک ورودی، $n$ نورون ReLU پنهان حداکثر $n$ نقطه شکست و در نتیجه حداکثر $n + 1$ تکه خطی می‌دهند. نورونی با $w_k = 0$ ثابت است ($\\relu(b_k)$) و فقط خروجی را جابه‌جا می‌کند. دو نورون می‌توانند همدیگر را خنثی کنند: $\\relu(x) - \\relu(-x) = x$ هیچ خمی ندارد، ولی $\\relu(x) + \\relu(-x) = |x|$ یک خم دارد.',
            )}
          </P>
          <P>
            {bi(
              'Any continuous piecewise-linear function of one variable with $m$ bends can be written exactly with $m$ such neurons plus a linear term, which is the 1-D core of the universal approximation theorem. "Best with 8" is the piecewise-linear interpolant of $0.5x^2$ with knots at $0, \\pm0.5, \\pm1, \\pm1.5$.',
              'هر تابع پیوسته تکه‌ای‌خطی یک‌متغیره با $m$ خم را می‌شود دقیقاً با $m$ نورون از این نوع به‌علاوه یک جمله خطی نوشت؛ این هسته یک‌بعدی قضیه تقریب‌زنی جهانی است. «بهترین با ۸» همان درون‌یاب تکه‌ای‌خطی $0.5x^2$ با گره‌های $0, \\pm0.5, \\pm1, \\pm1.5$ است.',
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
  const m = useModel(s);
  const selU = s.units[Math.min(s.sel, s.n - 1)];
  const selBp = breakpoint(selU);
  const layers = useMemo<PlotLayer[]>(() => {
    const out: PlotLayer[] = [];
    m.used.forEach((u, k) => {
      const bp = breakpoint(u);
      if (bp !== null && bp > XD[0] && bp < XD[1]) out.push({ kind: 'vline', at: bp, color: k === s.sel ? 'w' : 'var(--line-strong)', label: String(k + 1) });
    });
    if (s.parts) {
      m.used.forEach((u, k) => {
        if (k !== s.sel) out.push({ kind: 'fn', f: (x) => contribution(u, x), color: pieceColor(u), width: 1.4, opacity: 0.55 });
      });
      if (s.sel < s.n) out.push({ kind: 'fn', f: (x) => contribution(selU, x), color: pieceColor(selU), width: 2.8 });
    }
    out.push(
      { kind: 'points', color: 'y', pts: DATA.X.map((x, i) => ({ x: x[0], y: DATA.Y[i][0], r: 4.5 })) },
      { kind: 'fn', f: m.f, color: 'yhat', width: 3.2 },
    );
    return out;
  }, [m, s.sel, s.n, s.parts, selU]);
  return (
    <div className="viz">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div className="legend-chips">
          <span className="chip">
            <i style={{ background: 'var(--c-yhat)' }} /> {tr(bi('network output ŷ', 'خروجی شبکه ŷ'))}
          </span>
          <span className="chip">
            <i style={{ background: 'var(--c-y)' }} /> {tr(bi('target 0.5x²', 'هدف 0.5x²'))}
          </span>
          <span className="chip">
            <i style={{ background: 'var(--c-wpos)' }} /> {tr(bi('piece with v > 0', 'تکه با v > 0'))}
          </span>
          <span className="chip">
            <i style={{ background: 'var(--c-wneg)' }} /> {tr(bi('piece with v < 0', 'تکه با v < 0'))}
          </span>
        </div>
        <label className="small row" style={{ gap: 4 }}>
          <input type="checkbox" checked={s.parts} onChange={(e) => set({ parts: e.target.checked })} />
          {tr(bi('Show each neuron\'s piece', 'نمایش تکه هر نورون'))}
        </label>
      </div>
      <Plot2D
        testId="neurons-plot"
        xDomain={XD}
        yDomain={[-2.5, 5]}
        layers={layers}
        height={360}
        xLabel={tr(bi('input x', 'ورودی x'))}
        yLabel="ŷ"
        handle={selBp !== null && selBp > XD[0] && selBp < XD[1] ? { x: selBp, y: m.f(selBp), color: 'w' } : undefined}
        onDrag={(x) => {
          if (Math.abs(selU.w) < 1e-9) return;
          const bp = clamp(Math.round(x * 20) / 20, XD[0], XD[1]);
          const units = s.units.slice();
          units[s.sel] = { ...selU, b: Math.round(-selU.w * bp * 1000) / 1000 };
          set({ units });
        }}
        ariaLabel="hidden neuron pieces and their sum"
      />
      <div className="row">
        <span className="small muted">{tr(bi('Neuron:', 'نورون:'))}</span>
        {m.used.map((u, k) => (
          <button
            key={k}
            className={`btn sm ${k === s.sel ? 'active' : ''}`}
            style={{ borderColor: pieceColor(u).startsWith('var') ? pieceColor(u) : `var(--c-${pieceColor(u)})` }}
            onClick={() => set({ sel: k })}
            data-testid={`neuron-${k + 1}`}
          >
            {k + 1}
          </button>
        ))}
      </div>
      <p className="small muted">{tr(bi('Drag on the plot to move the selected neuron\'s breakpoint (it changes b).', 'روی نمودار بکش تا نقطه شکست نورون انتخاب‌شده جابه‌جا شود (b عوض می‌شود).'))}</p>
    </div>
  );
}

const SCEN: { id: ScenarioId; label: ReturnType<typeof bi> }[] = [
  { id: 'preset', label: bi('Start (preset)', 'شروع (پیش‌تنظیم)') },
  { id: 'fit4', label: bi('Best with 4', 'بهترین با ۴') },
  { id: 'fit8', label: bi('Best with 8', 'بهترین با ۸') },
  { id: 'duplicate', label: bi('8 = 4 duplicated', '۸ = ۴ تکراری') },
  { id: 'offscreen', label: bi('Off-screen breakpoints', 'شکست بیرون از دید') },
  { id: 'zeroV', label: bi('Output weights 0', 'وزن خروجی ۰') },
];

function Right() {
  const { tr } = useT();
  const [s, set] = useS();
  const m = useModel(s);
  const k = Math.min(s.sel, s.n - 1);
  const u = s.units[k];
  const bp = breakpoint(u);
  const side = activeSide(u);
  const setUnit = (p: Partial<Unit>) => {
    const units = s.units.slice();
    units[k] = { ...u, ...p };
    set({ units });
  };
  const K = k + 1;
  return (
    <div className="stack">
      <Slider label={tr(bi('hidden neurons n', 'نورون‌های پنهان n'))} value={s.n} min={1} max={MAX_UNITS} step={1} digits={0} onChange={(n) => set({ n: Math.round(n), sel: Math.min(s.sel, Math.round(n) - 1) })} testId="l04-n" />
      <div className="row">
        {SCEN.map((c) => (
          <button key={c.id} className="btn sm" onClick={() => set({ ...scenario(c.id), sel: 0 })} data-testid={`scen-${c.id}`}>
            {tr(c.label)}
          </button>
        ))}
      </div>
      <div className="stat-grid">
        <Stat k={tr(bi('neurons', 'نورون‌ها'))} v={s.n} />
        <Stat k={tr(bi('bends on screen', 'خم‌های روی صفحه'))} v={m.bends.length} testId="l04-bends" />
        <Stat k={tr(bi('straight pieces', 'تکه‌های صاف'))} v={m.bends.length + 1} />
        <Stat k="MSE" v={fmt(m.mse, 4)} color="L" testId="l04-mse" />
      </div>
      <h4>{tr(bi(`Neuron ${K}`, `نورون ${K}`))}</h4>
      <Slider label={`w${K}`} value={u.w} min={-3} max={3} step={0.05} color="w" onChange={(w) => setUnit({ w })} testId="l04-w" />
      <Slider label={`b${K}`} value={u.b} min={-4} max={4} step={0.05} color="b" onChange={(b) => setUnit({ b })} testId="l04-b" />
      <Slider label={`v${K}`} value={u.v} min={-2} max={2} step={0.05} color="w" onChange={(v) => setUnit({ v })} testId="l04-v" />
      <Tex display>{`\\text{piece}_{${K}}(x) = \\cw{${tex(u.v, 3)}}\\,\\relu(\\cw{${tex(u.w, 3)}}\\,\\cx{x} + \\cb{${texP(u.b, 3)}})`}</Tex>
      {bp !== null ? (
        <Tex display>{`\\text{breakpoint: } x = -\\frac{b_{${K}}}{w_{${K}}} = -\\frac{${tex(u.b, 3)}}{${texP(u.w, 3)}} = ${tex(bp, 3)}`}</Tex>
      ) : (
        <P className="small">{bi('$w = 0$: no breakpoint. The neuron outputs the constant $\\relu(b)$ for every $x$.', '$w = 0$: نقطه شکستی وجود ندارد. نورون برای هر $x$ مقدار ثابت $\\relu(b)$ را می‌دهد.')}</P>
      )}
      <P className="small muted">
        {side === 'right'
          ? bi('On (sloped) right of the breakpoint, off (0) left of it.', 'سمت راست نقطه شکست روشن (شیب‌دار) و سمت چپ آن خاموش (۰) است.')
          : side === 'left'
            ? bi('On (sloped) left of the breakpoint, off (0) right of it.', 'سمت چپ نقطه شکست روشن (شیب‌دار) و سمت راست آن خاموش (۰) است.')
            : side === 'always'
              ? bi('Always on with a constant output.', 'همیشه روشن با خروجی ثابت.')
              : bi('Always off: it contributes 0 everywhere.', 'همیشه خاموش: همه‌جا ۰ اضافه می‌کند.')}
      </P>
      <P className="small muted">{bi(`Slope change at its bend: $v\\,|w| = ${tex(u.v * Math.abs(u.w), 3)}$.`, `تغییر شیب در خمش: $v\\,|w| = ${tex(u.v * Math.abs(u.w), 3)}$.`)}</P>
      <Slider label={tr(bi('output bias c', 'بایاس خروجی c'))} value={s.c} min={-2} max={2} step={0.05} color="b" onChange={(c) => set({ c })} />
    </div>
  );
}

function Bottom() {
  const { tr } = useT();
  const [s, set] = useS();
  const m = useModel(s);
  const x = s.probe;
  const parts = m.used.map((u) => contribution(u, x));
  const sum = parts.reduce((a, b) => a + b, s.c);
  const engine = m.f(x);
  return (
    <div className="stack">
      <Slider label={tr(bi('probe input x', 'ورودی آزمایشی x'))} value={x} min={XD[0]} max={XD[1]} step={0.05} color="x" onChange={(probe) => set({ probe })} testId="l04-probe" />
      <table className="tbl" data-testid="neuron-table">
        <thead>
          <tr>
            <th>k</th>
            <th className="num c-w">w</th>
            <th className="num c-b">b</th>
            <th className="num c-w">v</th>
            <th className="num">{tr(bi('breakpoint', 'نقطه شکست'))}</th>
            <th className="num c-z">z = w·x + b</th>
            <th className="num c-a">a = ReLU(z)</th>
            <th className="num">v·a</th>
          </tr>
        </thead>
        <tbody>
          {m.used.map((u, k) => {
            const z = u.w * x + u.b;
            const bp = breakpoint(u);
            return (
              <tr key={k} className={k === s.sel ? 'hl' : ''} onClick={() => set({ sel: k })} style={{ cursor: 'pointer' }}>
                <td>{k + 1}</td>
                <td className="num">{fmt(u.w, 3)}</td>
                <td className="num">{fmt(u.b, 3)}</td>
                <td className="num">{fmt(u.v, 3)}</td>
                <td className="num">{bp === null ? '—' : fmt(bp, 3)}</td>
                <td className="num">{fmt(z, 3)}</td>
                <td className={`num ${z > 0 ? '' : 'muted'}`}>{fmt(act('relu', z), 3)}</td>
                <td className="num">{fmtSigned(parts[k], 3)}</td>
              </tr>
            );
          })}
          <tr>
            <td colSpan={7}>{tr(bi('output bias c', 'بایاس خروجی c'))}</td>
            <td className="num c-b">{fmtSigned(s.c, 3)}</td>
          </tr>
          <tr className="hl">
            <td colSpan={7}>
              <strong>{tr(bi('sum of pieces + c', 'جمع تکه‌ها + c'))}</strong>
            </td>
            <td className="num c-yhat" data-testid="l04-sum">
              {fmt(sum, 4)}
            </td>
          </tr>
          <tr>
            <td colSpan={7}>{tr(bi('network output from the engine (predict)', 'خروجی شبکه از موتور (predict)'))}</td>
            <td className="num c-yhat" data-testid="l04-engine">
              {fmt(engine, 4)}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export const lesson04: LessonDef = {
  id: ID,
  title: bi('Multiple Neurons', 'چند نورون'),
  short: bi('Multiple neurons', 'چند نورون'),
  objectives: [
    bi('See each hidden ReLU neuron as a hinge v·ReLU(w·x + b) with a breakpoint at x = −b/w', 'هر نورون ReLU پنهان را یک لولا v·ReLU(w·x + b) با نقطه شکست x = −b/w ببینی'),
    bi('Build a curved output by adding hinges, and predict how a negative output weight flips a piece', 'با جمع کردن لولاها خروجی منحنی بسازی و پیش‌بینی کنی وزن خروجی منفی چطور یک تکه را وارونه می‌کند'),
    bi('Fit y = 0.5x² by hand with 4 and with 8 neurons', 'منحنی y = 0.5x² را با ۴ و با ۸ نورون با دست بپوشانی'),
    bi('Explain why more neurons raise capacity but do not guarantee a more complex or more accurate function', 'توضیح بدهی چرا نورون بیشتر ظرفیت را بالا می‌برد ولی تابع پیچیده‌تر یا دقیق‌تری را تضمین نمی‌کند'),
  ],
  revealSteps: 5,
  Explain,
  Center,
  Right,
  Bottom,
  checkpoint: {
    question: bi('You add 4 more hidden neurons, but all their output weights $v$ are 0. What happens to the network\'s output curve?', '۴ نورون پنهان دیگر اضافه می‌کنی، ولی وزن خروجی $v$ همه آنها ۰ است. منحنی خروجی شبکه چه می‌شود؟'),
    choices: [
      { id: 'a', label: bi('Nothing changes', 'هیچ تغییری نمی‌کند') },
      { id: 'b', label: bi('It gets 4 more bends', '۴ خم دیگر پیدا می‌کند') },
      { id: 'c', label: bi('It fits the data better', 'داده را بهتر می‌پوشاند') },
    ],
    answer: 'a',
    explain: bi(
      'Each new piece is $0 \\cdot \\relu(\\ldots) = 0$. Capacity went up, but the function is identical. Check it with the "Output weights 0" scenario.',
      'هر تکه جدید برابر $0 \\cdot \\relu(\\ldots) = 0$ است. ظرفیت بالا رفته ولی تابع دقیقاً همان است. با سناریوی «وزن خروجی ۰» بررسی‌اش کن.',
    ),
  },
};
