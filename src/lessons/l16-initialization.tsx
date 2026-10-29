// Lesson 16: Weight initialization. Part 1 trains the same 1→4→1 tanh network from five
// initializations (same seed, same data, real Adam steps) and tracks every hidden unit, so the
// symmetry of zero/constant init is visible. Part 2 measures activation spread per layer of a
// deep stack for small, large, Xavier and He initialization. Weights come from engine initialize().

import { useEffect, useMemo } from 'react';
import type { LessonDef } from './types';
import { bi, useT, type Bi } from '../i18n';
import { useLessonState } from '../state/lessonState';
import { Plot2D, type PlotLayer } from '../components/Plot2D';
import { Tex } from '../components/Tex';
import { Segmented, Slider, Stat } from '../components/controls';
import { Advanced, Callout, Cycle, P, Predict, Reveal } from '../components/teaching';
import { fmt, predict, tex, type InitName } from '../engine';
import { DEEP, DEEP_INITS, HIDDEN, SYM_DATA, deepStats, heStd, symNet, trainFrom, unitSpread, xavierR, type DeepInit, type InitRun } from './math/l16';

const ID = 'l16';
const INITS: InitName[] = ['zeros', 'constant', 'uniform', 'xavier', 'he'];
const INIT_COLOR: Record<InitName, string> = { zeros: 'var(--c-L)', constant: 'var(--c-wneg)', uniform: 'var(--c-b)', xavier: 'var(--c-x)', he: 'var(--c-yhat)' };
const UNIT_COLOR = ['var(--c-x)', 'var(--c-wpos)', 'var(--c-w)', 'var(--c-wneg)'];
const DEEP_NAME: Record<DeepInit, Bi> = {
  small: bi('U(−0.1, 0.1)', 'U(−0.1, 0.1)'),
  large: bi('U(−1.5, 1.5)', 'U(−1.5, 1.5)'),
  xavier: bi('Xavier', 'Xavier'),
  he: bi('He', 'He'),
};
const DEEP_COLOR: Record<DeepInit, string> = { small: 'var(--c-z)', large: 'var(--c-L)', xavier: 'var(--c-x)', he: 'var(--c-yhat)' };

interface S {
  view: 'symmetry' | 'depth';
  init: InitName;
  seed: number;
  constant: number;
  range: number;
  steps: number;
  step: number;
  playing: boolean;
  act: 'tanh' | 'relu';
  deep: DeepInit;
}
const INIT: S = { view: 'symmetry', init: 'constant', seed: 3, constant: 0.5, range: 1, steps: 300, step: 300, playing: false, act: 'tanh', deep: 'xavier' };
const useS = () => useLessonState<S>(ID, INIT);

/** Remembers the last result so the three panels share one computation. */
function memoLast<A extends (string | number)[], R>(fn: (...a: A) => R) {
  let key = '';
  let val: R;
  return (...a: A): R => {
    const k = a.join('|');
    if (k !== key) {
      val = fn(...a);
      key = k;
    }
    return val;
  };
}

/** All five initializations trained from the same seed (bounded: 5 × ≤ 600 Adam steps). */
const runsFor = memoLast((seed: number, constant: number, range: number, steps: number) => {
  const scaleOf = (i: InitName) => (i === 'constant' ? constant : i === 'uniform' ? range : 1);
  return Object.fromEntries(INITS.map((i) => [i, trainFrom(symNet(i, seed, scaleOf(i)), SYM_DATA, steps)])) as Record<InitName, InitRun>;
});
const deepFor = memoLast((act: 'tanh' | 'relu') => Object.fromEntries(DEEP_INITS.map((d) => [d, deepStats(act, d)])) as Record<DeepInit, ReturnType<typeof deepStats>>);

function useRuns() {
  const [s] = useS();
  return runsFor(s.seed, s.constant, s.range, s.steps);
}

function useDeep() {
  const [s] = useS();
  return deepFor(s.act);
}

const kOf = (r: InitRun, step: number) => Math.min(step, r.loss.length - 1);

function Explain() {
  const { tr } = useT();
  return (
    <>
      <Cycle active={1} />
      <Reveal at={1}>
        <h3>{tr(bi('1. Where training starts', '۱. آموزش از کجا شروع می‌شود'))}</h3>
        <P>
          {bi(
            'Gradient descent only walks downhill from wherever it starts. Initialization is the choice of those starting weights. Two things can go wrong: neurons that start identical, and weights whose size is badly scaled.',
            'گرادیان کاهشی فقط از همان جایی که شروع کرده سرازیر می‌شود. مقداردهی اولیه (initialization) یعنی انتخاب همین وزن‌های شروع. دو چیز ممکن است خراب شود: نورون‌هایی که یکسان شروع می‌کنند، و وزن‌هایی که اندازه‌شان درست نیست.',
          )}
        </P>
      </Reveal>
      <Reveal at={2}>
        <Predict
          id={ID + '-zeros'}
          question={bi('A 1→4→1 tanh network starts with every weight and bias at 0. What happens to the 4 hidden neurons during training?', 'یک شبکه 1→4→1 با tanh با همه وزن‌ها و بایاس‌های صفر شروع می‌کند. در طول آموزش چه بر سر ۴ نورون پنهان می‌آید؟')}
          choices={[
            { id: 'a', label: bi('They learn different features', 'ویژگی‌های متفاوتی یاد می‌گیرند') },
            { id: 'b', label: bi('They stay at 0, identical, forever', 'برای همیشه صفر و یکسان می‌مانند') },
            { id: 'c', label: bi('They become random', 'تصادفی می‌شوند') },
          ]}
          answer="b"
          explain={bi(
            'Pick "zeros": all four weight curves stay flat at 0 and the loss never moves. Each hidden output is $\\tanh(0) = 0$, so the output weights get no gradient, and the output weights are 0, so no gradient flows back to the hidden layer.',
            '«zeros» را انتخاب کن: هر چهار منحنی وزن روی ۰ صاف می‌مانند و خطا تکان نمی‌خورد. خروجی هر نورون پنهان $\\tanh(0) = 0$ است، پس وزن‌های خروجی گرادیانی نمی‌گیرند، و چون وزن‌های خروجی ۰ هستند، هیچ گرادیانی هم به لایه پنهان برنمی‌گردد.',
          )}
        />
      </Reveal>
      <Reveal at={3}>
        <h3>{tr(bi('2. Symmetry is never broken', '۲. تقارن هیچ‌وقت شکسته نمی‌شود'))}</h3>
        <P>
          {bi(
            'Now pick "constant" (every weight 0.5). The units do learn, but their curves lie exactly on top of each other. Same weights give the same output $a_j$, the same gradient, and therefore the same update, step after step. Four identical neurons are worth one neuron.',
            'حالا «constant» را انتخاب کن (همه وزن‌ها 0.5). نورون‌ها یاد می‌گیرند، ولی منحنی‌هایشان دقیقاً روی هم می‌افتند. وزن یکسان یعنی خروجی یکسان $a_j$، گرادیان یکسان و در نتیجه به‌روزرسانی یکسان، قدم به قدم. چهار نورون یکسان فقط به اندازه یک نورون می‌ارزند.',
          )}
        </P>
        <Tex display>{"\\cg{\\frac{\\partial L}{\\partial w_j}} = \\frac{\\partial L}{\\partial \\hat y}\\; v_j\\, \\tanh'(w_j x + b_j)\\; \\cx{x} \\quad\\text{is the same for every } j \\text{ if } (w_j, b_j, v_j) \\text{ are}"}</Tex>
        <P>
          {bi(
            'Random initialization (uniform, Xavier, He) gives each unit a different start, so each one receives a different gradient and they specialise. Compare the loss curves below.',
            'مقداردهی تصادفی (uniform، Xavier، He) به هر نورون شروع متفاوتی می‌دهد، پس هر کدام گرادیان متفاوتی می‌گیرد و هر کدام کار خودش را یاد می‌گیرد. منحنی‌های خطا را پایین مقایسه کن.',
          )}
        </P>
      </Reveal>
      <Reveal at={4}>
        <h3>{tr(bi('3. Scale matters with depth', '۳. با عمق، اندازه مهم می‌شود'))}</h3>
        <P>
          {bi(
            'Switch the view to "Depth". Each layer multiplies the signal by its weights. If the weights are too small the signal shrinks layer after layer until it vanishes; if too large, tanh units saturate at ±1 (where their slope is ≈ 0, so gradients die) and ReLU signals explode.',
            'نما را روی «عمق» بگذار. هر لایه سیگنال را در وزن‌هایش ضرب می‌کند. اگر وزن‌ها خیلی کوچک باشند، سیگنال لایه به لایه کوچک می‌شود تا محو شود؛ اگر خیلی بزرگ باشند، نورون‌های tanh در ±1 اشباع می‌شوند (آنجا شیبشان ≈ 0 است و گرادیان می‌میرد) و سیگنال ReLU منفجر می‌شود.',
          )}
        </P>
        <Tex display>{'\\text{Xavier: } w \\sim U(-r, r),\\ r = \\sqrt{\\frac{6}{n_{in}+n_{out}}} \\qquad \\text{He: } w \\sim \\mathcal N\\!\\left(0, \\sqrt{\\tfrac{2}{n_{in}}}\\right)'}</Tex>
        <Callout>
          {tr(
            bi(
              'Both scale the weights by the number of inputs (fan-in) so the spread of activations stays roughly constant from layer to layer. Xavier suits tanh/sigmoid; He suits ReLU, which zeroes half its inputs.',
              'هر دو، وزن‌ها را بر اساس تعداد ورودی‌ها (fan-in) مقیاس می‌کنند تا پراکندگی فعال‌سازی‌ها از لایه‌ای به لایه دیگر تقریباً ثابت بماند. Xavier برای tanh/sigmoid مناسب است؛ He برای ReLU که نصف ورودی‌هایش را صفر می‌کند.',
            ),
          )}
        </Callout>
      </Reveal>
      <Reveal at={5}>
        <Advanced title={bi('The variance argument', 'استدلال واریانس')}>
          <P>
            {bi(
              'For $z_j = \\sum_{i=1}^{n_{in}} w_{ji} a_i$ with independent zero-mean weights, $\\operatorname{Var}(z_j) = n_{in}\\operatorname{Var}(w)\\,\\mathbb E[a_i^2]$. Near 0, tanh is the identity, so keeping $\\operatorname{Var}(z)$ equal across layers needs $n_{in}\\operatorname{Var}(w) = 1$; keeping backward gradients steady needs $n_{out}\\operatorname{Var}(w) = 1$. Glorot & Bengio (2010) compromise with $\\operatorname{Var}(w) = 2/(n_{in}+n_{out})$; a uniform $U(-r,r)$ has variance $r^2/3$, giving $r = \\sqrt{6/(n_{in}+n_{out})}$.',
              'برای $z_j = \\sum_{i=1}^{n_{in}} w_{ji} a_i$ با وزن‌های مستقل با میانگین صفر، $\\operatorname{Var}(z_j) = n_{in}\\operatorname{Var}(w)\\,\\mathbb E[a_i^2]$. نزدیک صفر، tanh همانی است، پس برای ثابت ماندن $\\operatorname{Var}(z)$ بین لایه‌ها باید $n_{in}\\operatorname{Var}(w) = 1$ باشد؛ و برای ثابت ماندن گرادیان‌ها در مسیر برگشت $n_{out}\\operatorname{Var}(w) = 1$. Glorot و Bengio (۲۰۱۰) میانه را می‌گیرند: $\\operatorname{Var}(w) = 2/(n_{in}+n_{out})$؛ توزیع یکنواخت $U(-r,r)$ واریانس $r^2/3$ دارد، پس $r = \\sqrt{6/(n_{in}+n_{out})}$.',
            )}
          </P>
          <P>
            {bi(
              'With ReLU, $\\mathbb E[a^2] = \\tfrac12\\operatorname{Var}(z)$ (half the inputs are zeroed), so the forward condition becomes $\\operatorname{Var}(w) = 2/n_{in}$ (He et al., 2015). Biases can safely start at 0: the random weights already break the symmetry.',
              'با ReLU داریم $\\mathbb E[a^2] = \\tfrac12\\operatorname{Var}(z)$ (نصف ورودی‌ها صفر می‌شوند)، پس شرط مسیر رفت می‌شود $\\operatorname{Var}(w) = 2/n_{in}$ (He و همکاران، ۲۰۱۵). بایاس‌ها می‌توانند با خیال راحت از ۰ شروع کنند: وزن‌های تصادفی خودشان تقارن را می‌شکنند.',
            )}
          </P>
        </Advanced>
      </Reveal>
    </>
  );
}

function SymmetryView() {
  const { tr } = useT();
  const [s, set] = useS();
  const runs = useRuns();
  const run = runs[s.init];
  const k = kOf(run, s.step);
  const units = run.units[k];
  const spread = unitSpread(units);
  const wLayers = useMemo<PlotLayer[]>(
    () => [
      ...Array.from({ length: HIDDEN }, (_, j): PlotLayer => ({ kind: 'polyline', pts: run.units.slice(0, k + 1).map((u, t) => [t, u[j].w]), color: UNIT_COLOR[j], width: 2.4 - j * 0.35 })),
      { kind: 'vline', at: k, color: 'var(--muted)' },
    ],
    [run, k],
  );
  const fitLayers = useMemo<PlotLayer[]>(() => {
    const net = run.nets[k];
    return [
      ...units.map((u, j): PlotLayer => ({ kind: 'fn', f: (x) => u.v * Math.tanh(u.w * x + u.b), color: UNIT_COLOR[j], width: 1.3, dash: '5 4', opacity: 0.9 })),
      { kind: 'fn', f: (x) => predict(net, [x])[0], color: 'yhat', width: 2.6 },
      { kind: 'points', color: 'y', pts: SYM_DATA.X.map((x, n) => ({ x: x[0], y: SYM_DATA.Y[n][0], r: 4.5 })) },
    ];
  }, [run, k, units]);
  return (
    <>
      <Segmented<InitName> label="init" value={s.init} onChange={(init) => set({ init })} options={INITS.map((i) => ({ value: i, label: i }))} />
      <div className="viz-grid">
        <div>
          <div className="small muted">{tr(bi('Input weight wⱼ of each hidden unit vs training step', 'وزن ورودی wⱼ هر نورون پنهان بر حسب قدم آموزش'))}</div>
          <Plot2D testId="init-weights" xDomain={[0, s.steps]} yDomain="auto" layers={wLayers} height={220} xLabel={tr(bi('step', 'قدم'))} yLabel="wⱼ" />
        </div>
        <div>
          <div className="small muted">{tr(bi(`Fit at step ${k}: ŷ (green) = sum of the unit curves vⱼ·tanh(wⱼx + bⱼ) + b`, `برازش در قدم ${k}: ŷ (سبز) = جمع منحنی نورون‌ها vⱼ·tanh(wⱼx + bⱼ) + b`))}</div>
          <Plot2D testId="init-fit" xDomain={[-2.3, 2.3]} yDomain={[-2, 2]} layers={fitLayers} height={220} xLabel="x" yLabel="ŷ" />
        </div>
      </div>
      <div className="legend-chips">
        {units.map((_, j) => (
          <span key={j} className="chip">
            <i style={{ background: UNIT_COLOR[j] }} /> {tr(bi(`unit ${j + 1}`, `نورون ${j + 1}`))}
          </span>
        ))}
      </div>
      <Callout tone={spread < 1e-9 ? 'warn' : 'info'}>
        <span data-testid="init-spread">
          {spread < 1e-9
            ? tr(bi(`Largest difference between any two units: ${fmt(spread, 6)}. They are exact copies (you see only one curve).`, `بیشترین اختلاف بین هر دو نورون: ${fmt(spread, 6)}. کپی دقیق هم‌اند (فقط یک منحنی می‌بینی).`))
            : tr(bi(`Largest difference between any two units: ${fmt(spread, 4)}. Symmetry is broken: each unit learns its own piece.`, `بیشترین اختلاف بین هر دو نورون: ${fmt(spread, 4)}. تقارن شکسته شده: هر نورون تکه خودش را یاد می‌گیرد.`))}
        </span>
      </Callout>
    </>
  );
}

function DepthView() {
  const { tr } = useT();
  const [s, set] = useS();
  const deep = useDeep();
  const stdLayers = useMemo<PlotLayer[]>(
    () => DEEP_INITS.map((d): PlotLayer => ({ kind: 'polyline', pts: deep[d].map((st, l) => [l + 1, Math.log10(Math.max(st.std, 1e-6))]), color: DEEP_COLOR[d], width: d === s.deep ? 3.2 : 1.6, markers: true })),
    [deep, s.deep],
  );
  const sel = deep[s.deep];
  return (
    <>
      <div className="row">
        <Segmented<'tanh' | 'relu'> label="activation" value={s.act} onChange={(act) => set({ act })} options={[{ value: 'tanh', label: 'tanh' }, { value: 'relu', label: 'ReLU' }]} />
        <Segmented<DeepInit> label="deep init" value={s.deep} onChange={(deep) => set({ deep })} options={DEEP_INITS.map((d) => ({ value: d, label: tr(DEEP_NAME[d]) }))} />
      </div>
      <div>
        <div className="small muted">{tr(bi(`Std of activations per layer (log₁₀), ${DEEP.samples} random inputs through ${DEEP.depth} layers of ${DEEP.width} ${s.act} units`, `انحراف معیار فعال‌سازی‌ها در هر لایه (log₁₀)، ${DEEP.samples} ورودی تصادفی از ${DEEP.depth} لایه ${DEEP.width}تایی ${s.act}`))}</div>
        <Plot2D testId="init-depth-std" xDomain={[0.7, DEEP.depth + 0.3]} yDomain={[-5, 3]} layers={[{ kind: 'hline', at: 0, color: 'var(--muted)', label: 'std = 1' }, ...stdLayers]} height={210} xLabel={tr(bi('layer', 'لایه'))} yLabel="log₁₀ std" legend={DEEP_INITS.map((d) => ({ label: tr(DEEP_NAME[d]), color: DEEP_COLOR[d] }))} />
      </div>
      <div className="small muted">{tr(bi(`Histograms of the activations, init ${tr(DEEP_NAME[s.deep])}`, `هیستوگرام فعال‌سازی‌ها، مقداردهی ${tr(DEEP_NAME[s.deep])}`))}</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 8 }} data-testid="init-hists">
        {sel.map((st, l) => {
          const w = (st.hist.hi - st.hist.lo) / st.hist.counts.length;
          const top = Math.max(...st.hist.counts, 1e-9);
          return (
            <div key={l}>
              <div className="small">
                {tr(bi('layer', 'لایه'))} {l + 1} · std <span className="num">{fmt(st.std, 3)}</span>
              </div>
              <Plot2D
                xDomain={[st.hist.lo, st.hist.hi]}
                yDomain={[0, top * 1.1]}
                height={110}
                layers={st.hist.counts.map((c, i): PlotLayer => ({ kind: 'segment', from: [st.hist.lo + (i + 0.5) * w, 0], to: [st.hist.lo + (i + 0.5) * w, c], color: DEEP_COLOR[s.deep], width: 5 }))}
              />
            </div>
          );
        })}
      </div>
    </>
  );
}

function Center() {
  const { tr } = useT();
  const [s, set] = useS();
  useEffect(() => {
    if (!s.playing) return;
    const inc = Math.max(1, Math.round(s.steps / 75));
    const id = window.setInterval(() => set((q) => (q.step >= q.steps ? { playing: false } : { step: Math.min(q.steps, q.step + inc) })), 60);
    return () => window.clearInterval(id);
  }, [s.playing, s.steps, set]);
  return (
    <div className="viz">
      <Segmented
        label="view"
        value={s.view}
        onChange={(view) => set({ view })}
        options={[
          { value: 'symmetry', label: tr(bi('Symmetry (1→4→1 tanh)', 'تقارن (1→4→1 tanh)')) },
          { value: 'depth', label: tr(bi('Depth (6 layers)', 'عمق (۶ لایه)')) },
        ]}
      />
      {s.view === 'symmetry' ? <SymmetryView /> : <DepthView />}
      {s.view === 'symmetry' && (
        <div className="row">
          <button className={`btn ${s.playing ? '' : 'primary'}`} onClick={() => set({ playing: !s.playing, step: !s.playing && s.step >= s.steps ? 0 : s.step })} data-testid="init-play">
            {s.playing ? `❚❚ ${tr(bi('Pause', 'توقف'))}` : `▶ ${tr(bi('Play training', 'پخش آموزش'))}`}
          </button>
          <div className="grow">
            <Slider label={tr(bi('step', 'قدم'))} value={s.step} min={0} max={s.steps} step={1} digits={0} onChange={(step) => set({ step, playing: false })} testId="init-step" />
          </div>
        </div>
      )}
    </div>
  );
}

function Right() {
  const { tr } = useT();
  const [s, set] = useS();
  const runs = useRuns();
  const deep = useDeep();
  if (s.view === 'depth') {
    const n = DEEP.width;
    return (
      <div className="stack">
        <Tex display>{`\\text{Xavier: } r = \\sqrt{\\tfrac{6}{${n} + ${n}}} = ${tex(xavierR(n, n), 4)}`}</Tex>
        <Tex display>{`\\text{He: } \\sigma = \\sqrt{\\tfrac{2}{${n}}} = ${tex(heStd(n), 4)}`}</Tex>
        <P className="small muted">{bi(`Variance of $U(-1.5, 1.5)$ is $0.75$, so $n_{in}\\operatorname{Var}(w) = ${n} \\cdot 0.75 = ${tex(n * 0.75)}$: every layer amplifies.`, `واریانس $U(-1.5, 1.5)$ برابر $0.75$ است، پس $n_{in}\\operatorname{Var}(w) = ${n} \\cdot 0.75 = ${tex(n * 0.75)}$: هر لایه سیگنال را بزرگ می‌کند.`)}</P>
        <table className="tbl" data-testid="init-std-table">
          <thead>
            <tr>
              <th>{tr(bi('layer', 'لایه'))}</th>
              {DEEP_INITS.map((d) => (
                <th key={d} className="num" style={{ color: DEEP_COLOR[d] }}>{tr(DEEP_NAME[d])}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {deep.xavier.map((_, l) => (
              <tr key={l}>
                <td>{l + 1}</td>
                {DEEP_INITS.map((d) => (
                  <td key={d} className="num">{fmt(deep[d][l].std, 3)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        <div className="stat-grid">
          <Stat
            k={s.act === 'tanh' ? tr(bi('saturated |a| > 0.95, last layer', 'اشباع |a| > 0.95، لایه آخر')) : tr(bi('exactly 0, last layer', 'دقیقاً صفر، لایه آخر'))}
            v={`${fmt(100 * deep[s.deep][DEEP.depth - 1].extreme, 1)}%`}
          />
        </div>
        {s.act === 'relu' && <P className="small muted">{bi('About half of ReLU outputs are 0 by design; that alone is fine. The problem is when the non-zero half shrinks or explodes.', 'حدود نصف خروجی‌های ReLU طبیعتاً صفرند و این به‌تنهایی مشکلی نیست. مشکل وقتی است که نیمه غیرصفر کوچک یا منفجر شود.')}</P>}
      </div>
    );
  }
  const run = runs[s.init];
  const k = kOf(run, s.step);
  return (
    <div className="stack">
      <Slider label={tr(bi('seed', 'بذر (seed)'))} value={s.seed} min={1} max={30} step={1} digits={0} onChange={(seed) => set({ seed })} testId="init-seed" />
      <Slider label={tr(bi('constant value', 'مقدار ثابت'))} value={s.constant} min={-1} max={1} step={0.05} color="w" onChange={(constant) => set({ constant, init: 'constant' })} />
      <Slider label={tr(bi('uniform range ±', 'بازه یکنواخت ±'))} value={s.range} min={0.05} max={3} step={0.05} color="w" onChange={(range) => set({ range, init: 'uniform' })} />
      <Slider label={tr(bi('training steps', 'قدم‌های آموزش'))} value={s.steps} min={50} max={600} step={50} digits={0} onChange={(steps) => set({ steps, step: steps })} />
      <h4>{tr(bi('Scales for this network', 'مقیاس‌ها برای این شبکه'))}</h4>
      <Tex display>{`\\text{Xavier } r: \\ \\sqrt{\\tfrac{6}{1+4}} = ${tex(xavierR(1, 4), 3)}\\ \\text{(both layers)}`}</Tex>
      <Tex display>{`\\text{He } \\sigma: \\ \\sqrt{\\tfrac{2}{1}} = ${tex(heStd(1), 3)},\\ \\ \\sqrt{\\tfrac{2}{4}} = ${tex(heStd(4), 3)}`}</Tex>
      <h4>
        {tr(bi('Hidden units at step', 'نورون‌های پنهان در قدم'))} {k} ({s.init})
      </h4>
      <table className="tbl" data-testid="init-units">
        <thead>
          <tr>
            <th>j</th>
            <th className="num c-w">wⱼ</th>
            <th className="num c-b">bⱼ</th>
            <th className="num c-w">vⱼ</th>
          </tr>
        </thead>
        <tbody>
          {run.units[k].map((u, j) => (
            <tr key={j}>
              <td style={{ color: UNIT_COLOR[j] }}>{j + 1}</td>
              <td className="num">{fmt(u.w, 4)}</td>
              <td className="num">{fmt(u.b, 4)}</td>
              <td className="num">{fmt(u.v, 4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <P className="small muted">{bi('$w_j$: input → unit $j$, $b_j$: its bias, $v_j$: unit $j$ → output. Trained with Adam ($\\eta = 0.05$), full batch, on $y = \\sin(1.5x)$.', '$w_j$: ورودی ← نورون $j$، $b_j$: بایاس آن، $v_j$: نورون $j$ ← خروجی. آموزش با Adam ($\\eta = 0.05$)، کل داده، روی $y = \\sin(1.5x)$.')}</P>
    </div>
  );
}

function Bottom() {
  const { tr } = useT();
  const [s, set] = useS();
  const runs = useRuns();
  const layers = useMemo<PlotLayer[]>(
    () => [
      ...INITS.map((i): PlotLayer => ({ kind: 'polyline', pts: runs[i].loss.slice(0, kOf(runs[i], s.step) + 1).map((L, t) => [t, Math.log10(Math.max(L, 1e-8))]), color: INIT_COLOR[i], width: i === s.init ? 3 : 1.6 })),
      { kind: 'vline', at: s.step, color: 'var(--muted)' },
    ],
    [runs, s.step, s.init],
  );
  return (
    <div className="viz-grid">
      <div>
        <div className="small muted">{tr(bi(`Training loss for each initialization (same seed ${s.seed}, same data, log₁₀)`, `خطای آموزش برای هر مقداردهی (بذر یکسان ${s.seed}، داده یکسان، log₁₀)`))}</div>
        <Plot2D testId="init-loss" xDomain={[0, s.steps]} yDomain={[-4, 0.5]} layers={layers} height={210} xLabel={tr(bi('step', 'قدم'))} yLabel="log₁₀ L" legend={INITS.map((i) => ({ label: i, color: INIT_COLOR[i] }))} />
      </div>
      <table className="tbl">
        <thead>
          <tr>
            <th>init</th>
            <th className="num">{tr(bi('loss at step 0', 'خطا در قدم ۰'))}</th>
            <th className="num">{tr(bi(`loss at step ${s.step}`, `خطا در قدم ${s.step}`))}</th>
            <th className="num">{tr(bi('unit spread', 'اختلاف نورون‌ها'))}</th>
          </tr>
        </thead>
        <tbody>
          {INITS.map((i) => {
            const k = kOf(runs[i], s.step);
            return (
              <tr key={i} className={i === s.init ? 'hl' : ''} onClick={() => set({ init: i, view: 'symmetry' })} style={{ cursor: 'pointer' }}>
                <td style={{ color: INIT_COLOR[i], fontWeight: 600 }}>{i}</td>
                <td className="num">{fmt(runs[i].loss[0], 4)}</td>
                <td className="num c-L">{fmt(runs[i].loss[k], 4)}</td>
                <td className="num">{fmt(unitSpread(runs[i].units[k]), 4)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export const lesson16: LessonDef = {
  id: ID,
  title: bi('Weight Initialization', 'مقداردهی اولیه وزن‌ها'),
  short: bi('Initialization', 'مقداردهی اولیه'),
  objectives: [
    bi('See why zero or constant initialization keeps hidden units identical forever', 'ببینی چرا مقداردهی صفر یا ثابت نورون‌های پنهان را برای همیشه یکسان نگه می‌دارد'),
    bi('Explain how random initialization breaks symmetry', 'توضیح بدهی مقداردهی تصادفی چطور تقارن را می‌شکند'),
    bi('Watch activations vanish, saturate or explode with depth', 'محو شدن، اشباع یا انفجار فعال‌سازی‌ها را با افزایش عمق ببینی'),
    bi('State the Xavier and He scales and when to use each', 'مقیاس Xavier و He و زمان استفاده از هر کدام را بیان کنی'),
  ],
  revealSteps: 5,
  Explain,
  Center,
  Right,
  Bottom,
  checkpoint: {
    question: bi('Why can\'t all weights of a layer start at the same value?', 'چرا همه وزن‌های یک لایه نمی‌توانند از یک مقدار شروع کنند؟'),
    choices: [
      { id: 'a', label: bi('The loss would be undefined', 'خطا تعریف نشده می‌شود') },
      { id: 'b', label: bi('Identical units get identical gradients, so they stay copies of each other', 'نورون‌های یکسان گرادیان یکسان می‌گیرند، پس کپی هم باقی می‌مانند') },
      { id: 'c', label: bi('The learning rate must then be zero', 'آن وقت نرخ یادگیری باید صفر باشد') },
    ],
    answer: 'b',
    explain: bi(
      'Same weights → same activations → same gradients → same updates. With "constant", the four weight curves lie on top of each other at every step and the unit spread stays 0.',
      'وزن یکسان → فعال‌سازی یکسان → گرادیان یکسان → به‌روزرسانی یکسان. با «constant» چهار منحنی وزن در هر قدم روی هم می‌افتند و اختلاف نورون‌ها ۰ می‌ماند.',
    ),
  },
};
