// Lesson 15: Optimizers. SGD, SGD with momentum and Adam start from the same point on the same
// 2-D objective (a narrow valley or a curved valley). Every path is produced by the engine's
// optimizerStep; the lesson also keeps the optimizer state so the update equations show real numbers.

import { useEffect, useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import type { LessonDef } from './types';
import { bi, useT, type Bi } from '../i18n';
import { useLessonState } from '../state/lessonState';
import { Plot2D, type PlotLayer } from '../components/Plot2D';
import { PlotlyChart, cssVar, useThemeKey } from '../components/PlotlyChart';
import { Tex } from '../components/Tex';
import { Segmented, Slider } from '../components/controls';
import { Advanced, Callout, Cycle, P, Predict, Reveal } from '../components/teaching';
import { defaultOptimizer, fmt, tex, type OptimizerName } from '../engine';
import { contourGrid, makeObjective, tracePath, type ObjId, type Trace } from './math/l15';

const ID = 'l15';
const OPTS: OptimizerName[] = ['sgd', 'momentum', 'adam'];
const COLOR_VAR: Record<OptimizerName, string> = { sgd: '--c-wneg', momentum: '--c-x', adam: '--c-yhat' };
const NAME: Record<OptimizerName, Bi> = { sgd: bi('SGD', 'SGD'), momentum: bi('Momentum', 'Momentum'), adam: bi('Adam', 'Adam') };
const DEFAULT_LR: Record<ObjId, Record<OptimizerName, number>> = {
  valley: { sgd: 0.18, momentum: 0.04, adam: 0.3 },
  curved: { sgd: 0.005, momentum: 0.002, adam: 0.05 },
};
const DEFAULT_STEPS: Record<ObjId, number> = { valley: 60, curved: 300 };

interface S {
  obj: ObjId;
  a: number;
  b: number;
  c: number;
  lr: Record<OptimizerName, number>;
  beta: number;
  beta1: number;
  beta2: number;
  steps: number;
  step: number;
  playing: boolean;
  show: Record<OptimizerName, boolean>;
  focus: OptimizerName;
}
const INIT: S = {
  obj: 'valley',
  a: 0.4,
  b: 10,
  c: 10,
  lr: DEFAULT_LR.valley,
  beta: 0.9,
  beta1: 0.9,
  beta2: 0.999,
  steps: DEFAULT_STEPS.valley,
  step: DEFAULT_STEPS.valley,
  playing: false,
  show: { sgd: true, momentum: true, adam: true },
  focus: 'adam',
};
const useS = () => useLessonState<S>(ID, INIT);

function useTraces() {
  const [s] = useS();
  return useMemo(() => {
    const obj = makeObjective(s.obj, { a: s.a, b: s.b, c: s.c });
    const cfg = (name: OptimizerName) => ({ ...defaultOptimizer(name, s.lr[name]), momentum: s.beta, beta1: s.beta1, beta2: s.beta2 });
    const traces = Object.fromEntries(OPTS.map((o) => [o, tracePath(obj, cfg(o), s.steps)])) as Record<OptimizerName, Trace>;
    return { obj, traces };
  }, [s.obj, s.a, s.b, s.c, s.lr, s.beta, s.beta1, s.beta2, s.steps]);
}

const kOf = (t: Trace, step: number) => Math.min(step, t.points.length - 1);
const v2 = (v: number[], d = 3) => `(${tex(v[0], d)},\\ ${tex(v[1], d)})`;

function Explain() {
  const { tr } = useT();
  return (
    <>
      <Cycle active={1} />
      <Reveal at={1}>
        <h3>{tr(bi('1. The trouble with a narrow valley', '۱. دردسر یک دره باریک'))}</h3>
        <P>
          {bi(
            'The map shows $L(w_1, w_2) = \\tfrac12(a\\,w_1^2 + b\\,w_2^2)$ with $b$ much larger than $a$: steep walls across $w_2$, an almost flat floor along $w_1$. Plain gradient descent (SGD) must keep $\\eta < 2/b$ or it explodes across the steep direction (lesson 13), so along the flat floor it crawls, zig-zagging between the walls.',
            'نقشه $L(w_1, w_2) = \\tfrac12(a\\,w_1^2 + b\\,w_2^2)$ را نشان می‌دهد که در آن $b$ خیلی بزرگ‌تر از $a$ است: دیواره‌های تند در جهت $w_2$ و کفی تقریباً صاف در جهت $w_1$. گرادیان کاهشی ساده (SGD) باید $\\eta < 2/b$ را رعایت کند وگرنه در جهت تند منفجر می‌شود (درس ۱۳)، پس روی کف صاف دره فقط می‌خزد و بین دیواره‌ها زیگزاگ می‌زند.',
          )}
        </P>
      </Reveal>
      <Reveal at={2}>
        <h3>{tr(bi('2. Momentum: a ball with inertia', '۲. Momentum: توپی با اینرسی'))}</h3>
        <P>
          {bi(
            'Instead of stepping along the current gradient only, momentum keeps a velocity: a running sum of recent gradients. Pushes that keep pointing the same way (along the floor) add up and speed it up; pushes that flip sign every step (across the walls) cancel out.',
            'Momentum به جای قدم برداشتن فقط در جهت گرادیان فعلی، یک «سرعت» نگه می‌دارد: جمع جاری گرادیان‌های اخیر. فشارهایی که مدام در یک جهت‌اند (در طول کف دره) جمع می‌شوند و سرعت را بالا می‌برند؛ فشارهایی که هر قدم علامتشان عوض می‌شود (بین دیواره‌ها) همدیگر را خنثی می‌کنند.',
          )}
        </P>
      </Reveal>
      <Reveal at={3}>
        <h3>{tr(bi('3. Adam: a step size per parameter', '۳. Adam: اندازه قدم جدا برای هر پارامتر'))}</h3>
        <P>
          {bi(
            'Adam keeps, for every parameter separately, an average of its gradient and an average of its squared gradient. It divides the first by the square root of the second, so each parameter moves about $\\eta$ per step whether its gradient is huge or tiny. Steep and flat directions get similar step sizes.',
            'Adam برای هر پارامتر جداگانه یک میانگین از گرادیان و یک میانگین از مربع گرادیان نگه می‌دارد. اولی را بر جذر دومی تقسیم می‌کند، پس هر پارامتر در هر قدم حدود $\\eta$ جابه‌جا می‌شود، چه گرادیانش خیلی بزرگ باشد چه خیلی کوچک. جهت‌های تند و صاف اندازه قدم مشابهی می‌گیرند.',
          )}
        </P>
        <Predict
          id={ID + '-adam1'}
          question={bi('On the valley, start at step 0 and take one step. Which optimizer moves $w_1$ and $w_2$ by the same amount?', 'روی دره از قدم ۰ یک قدم بردار. کدام بهینه‌ساز $w_1$ و $w_2$ را به یک اندازه جابه‌جا می‌کند؟')}
          choices={OPTS.map((o) => ({ id: o, label: NAME[o] }))}
          answer="adam"
          explain={bi(
            'At $t = 1$, $\\hat m = g$ and $\\hat s = g^2$, so the step is $\\eta\\, g/|g| = \\eta$ in each coordinate: a diagonal move of $(+0.3, -0.3)$. SGD and momentum move by $\\eta g$, much more along steep $w_2$. Set the step slider to 1 and read the right panel.',
            'در $t = 1$ داریم $\\hat m = g$ و $\\hat s = g^2$، پس قدم در هر مختصات $\\eta\\, g/|g| = \\eta$ است: یک حرکت قطری $(+0.3, -0.3)$. SGD و Momentum به اندازه $\\eta g$ جابه‌جا می‌شوند که در جهت تند $w_2$ خیلی بیشتر است. اسلایدر قدم را روی ۱ بگذار و پنل راست را بخوان.',
          )}
        />
      </Reveal>
      <Reveal at={4}>
        <h3>{tr(bi('4. The exact updates', '۴. فرمول‌های دقیق'))}</h3>
        <P>{bi('With gradient $\\cg{g_t}$ at step $t$ (all operations per parameter):', 'با گرادیان $\\cg{g_t}$ در قدم $t$ (همه عملیات برای هر پارامتر جداگانه):')}</P>
        <Tex display>{'\\text{SGD:}\\quad \\theta \\leftarrow \\theta - \\clr{\\eta}\\,\\cg{g_t}'}</Tex>
        <Tex display>{'\\text{Momentum:}\\quad v \\leftarrow \\beta v + \\cg{g_t},\\qquad \\theta \\leftarrow \\theta - \\clr{\\eta}\\,v'}</Tex>
        <Tex display>{'\\text{Adam:}\\quad \\begin{aligned} m &\\leftarrow \\beta_1 m + (1-\\beta_1)\\cg{g_t} \\\\ s &\\leftarrow \\beta_2 s + (1-\\beta_2)\\cg{g_t}^2 \\\\ \\hat m &= \\frac{m}{1-\\beta_1^t},\\quad \\hat s = \\frac{s}{1-\\beta_2^t} \\\\ \\theta &\\leftarrow \\theta - \\clr{\\eta}\\,\\frac{\\hat m}{\\sqrt{\\hat s}+\\varepsilon} \\end{aligned}'}</Tex>
        <Callout>
          {tr(
            bi(
              'Switch to the curved valley. Its floor bends, so the "right" direction keeps changing. SGD needs a tiny learning rate to survive the steep start and then barely moves; momentum and Adam follow the bend.',
              'به دره خمیده برو. کف آن خم می‌شود، پس جهت «درست» مدام عوض می‌شود. SGD برای اینکه در شروعِ تند منفجر نشود نرخ یادگیری خیلی کوچکی لازم دارد و بعد تقریباً تکان نمی‌خورد؛ Momentum و Adam خم را دنبال می‌کنند.',
            ),
          )}
        </Callout>
      </Reveal>
      <Reveal at={5}>
        <Advanced title={bi('Why these formulas work', 'چرا این فرمول‌ها کار می‌کنند')}>
          <P>
            {bi(
              'Momentum: unrolling gives $v_t = \\sum_{j\\ge0} \\beta^j g_{t-j}$. For a constant gradient this tends to $g/(1-\\beta)$, so the effective learning rate along a consistent direction is $\\eta/(1-\\beta)$ (10× for $\\beta = 0.9$), while an alternating gradient $\\pm g$ sums to about $g/(1+\\beta)$. On a quadratic with condition number $\\kappa$, well-tuned heavy-ball momentum converges at rate $\\approx (\\sqrt\\kappa-1)/(\\sqrt\\kappa+1)$ per step instead of $(\\kappa-1)/(\\kappa+1)$.',
              'Momentum: با باز کردن رابطه $v_t = \\sum_{j\\ge0} \\beta^j g_{t-j}$ به دست می‌آید. برای گرادیان ثابت این به $g/(1-\\beta)$ میل می‌کند، پس نرخ یادگیری مؤثر در یک جهت ثابت $\\eta/(1-\\beta)$ است (برای $\\beta = 0.9$ ده برابر)، در حالی که گرادیان رفت‌وبرگشتی $\\pm g$ حدود $g/(1+\\beta)$ جمع می‌شود. روی تابع درجه دو با عدد حالت $\\kappa$، Momentum خوب‌تنظیم‌شده با نرخ $\\approx (\\sqrt\\kappa-1)/(\\sqrt\\kappa+1)$ در هر قدم همگرا می‌شود به جای $(\\kappa-1)/(\\kappa+1)$.',
            )}
          </P>
          <P>
            {bi(
              'Adam bias correction: with $m_0 = 0$, $m_t = (1-\\beta_1)\\sum_{j=1}^t \\beta_1^{t-j} g_j$, so for a stationary gradient $\\mathbb E[m_t] = (1-\\beta_1^t)\\,\\mathbb E[g]$. Dividing by $1-\\beta_1^t$ removes the pull toward 0 in early steps (same for $s$). $\\varepsilon = 10^{-8}$ only prevents division by zero. Adam is a strong default, not a guarantee: it still needs a tuned $\\eta$, and in practice AdamW (decoupled weight decay) is common.',
              'اصلاح سوگیری Adam: با $m_0 = 0$ داریم $m_t = (1-\\beta_1)\\sum_{j=1}^t \\beta_1^{t-j} g_j$، پس برای گرادیان ایستا $\\mathbb E[m_t] = (1-\\beta_1^t)\\,\\mathbb E[g]$. تقسیم بر $1-\\beta_1^t$ کشش به سمت صفر را در قدم‌های اول حذف می‌کند (برای $s$ هم همین‌طور). $\\varepsilon = 10^{-8}$ فقط جلوی تقسیم بر صفر را می‌گیرد. Adam یک پیش‌فرض قوی است نه تضمین: هنوز $\\eta$ تنظیم‌شده لازم دارد و در عمل AdamW (با weight decay جدا) رایج است.',
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
  const { obj, traces } = useTraces();
  const theme = useThemeKey();
  useEffect(() => {
    if (!s.playing) return;
    const inc = Math.max(1, Math.round(s.steps / 60));
    const id = window.setInterval(() => set((q) => (q.step >= q.steps ? { playing: false } : { step: Math.min(q.steps, q.step + inc) })), 70);
    return () => window.clearInterval(id);
  }, [s.playing, s.steps, set]);
  const grid = useMemo(() => contourGrid(obj), [obj]);
  const contour = useMemo<Data>(
    () =>
      ({
        type: 'contour',
        x: grid.xs,
        y: grid.ys,
        z: grid.z,
        ncontours: 24,
        contours: { coloring: 'lines' },
        line: { width: 1 },
        colorscale: [
          [0, cssVar('--c-a')],
          [1, cssVar('--line-strong')],
        ],
        showscale: false,
        hoverinfo: 'skip',
      }) as Data,
    // theme: re-resolve CSS colors when the theme changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [grid, theme],
  );
  const data = useMemo<Data[]>(() => {
    const out: Data[] = [contour];
    for (const o of OPTS) {
      if (!s.show[o]) continue;
      const t = traces[o];
      const k = kOf(t, s.step);
      const pts = t.points.slice(0, k + 1);
      const color = cssVar(COLOR_VAR[o]);
      out.push({
        type: 'scatter',
        mode: 'lines+markers',
        x: pts.map((p) => p[0]),
        y: pts.map((p) => p[1]),
        customdata: pts.map((_, i) => i),
        line: { color, width: 2 },
        marker: { color, size: 4 },
        hovertemplate: `${NAME[o].en}<br>k = %{customdata}<br>w₁ = %{x:.3f}<br>w₂ = %{y:.3f}<extra></extra>`,
      } as Data);
      out.push({ type: 'scatter', mode: 'markers', x: [pts[k][0]], y: [pts[k][1]], marker: { color, size: 12, line: { color: cssVar('--surface'), width: 2 } }, hoverinfo: 'skip' } as Data);
    }
    out.push({ type: 'scatter', mode: 'markers', x: [obj.start[0]], y: [obj.start[1]], marker: { color: cssVar('--ink-2'), size: 9, symbol: 'circle-open', line: { width: 2 } }, hoverinfo: 'skip' } as Data);
    out.push({ type: 'scatter', mode: 'markers', x: [obj.min[0]], y: [obj.min[1]], marker: { color: cssVar('--c-L'), size: 13, symbol: 'star' }, hoverinfo: 'skip' } as Data);
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [contour, traces, s.step, s.show, obj, theme]);
  const layout = useMemo<Partial<Layout>>(
    () => ({
      uirevision: `${obj.id}`,
      margin: { l: 46, r: 10, t: 10, b: 40 },
      xaxis: { range: obj.xRange, title: { text: 'w₁' } },
      yaxis: { range: obj.yRange, title: { text: 'w₂' } },
      hovermode: 'closest',
    }),
    [obj],
  );
  return (
    <div className="viz">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <Segmented<ObjId>
          label="objective"
          value={s.obj}
          onChange={(o) => set({ obj: o, lr: DEFAULT_LR[o], steps: DEFAULT_STEPS[o], step: 0, playing: false })}
          options={[
            { value: 'valley', label: tr(bi('Narrow valley', 'دره باریک')) },
            { value: 'curved', label: tr(bi('Curved valley', 'دره خمیده')) },
          ]}
        />
        <div className="legend-chips">
          {OPTS.map((o) => (
            <label key={o} className="chip" style={{ cursor: 'pointer', opacity: s.show[o] ? 1 : 0.45 }}>
              <input type="checkbox" checked={s.show[o]} onChange={(e) => set({ show: { ...s.show, [o]: e.target.checked } })} style={{ margin: 0 }} />
              <i style={{ background: `var(${COLOR_VAR[o]})` }} /> {tr(NAME[o])}
            </label>
          ))}
        </div>
      </div>
      <Tex className="formula-hero" display>
        {s.obj === 'valley' ? `\\cL{L} = \\tfrac12(${tex(s.a)}\\,w_1^2 + ${tex(s.b)}\\,w_2^2)` : `\\cL{L} = (1-w_1)^2 + ${tex(s.c)}\\,(w_2 - w_1^2)^2`}
      </Tex>
      <PlotlyChart data={data} layout={layout} height={380} testId="opt-contour" />
      <div className="row">
        <button className={`btn ${s.playing ? '' : 'primary'}`} onClick={() => set({ playing: !s.playing, step: !s.playing && s.step >= s.steps ? 0 : s.step })} data-testid="opt-play">
          {s.playing ? `❚❚ ${tr(bi('Pause', 'توقف'))}` : `▶ ${tr(bi('Play', 'پخش'))}`}
        </button>
        <div className="grow">
          <Slider label={tr(bi('step k', 'قدم k'))} value={s.step} min={0} max={s.steps} step={1} digits={0} onChange={(step) => set({ step, playing: false })} testId="opt-step" />
        </div>
      </div>
      <p className="small muted">{tr(bi('Hollow circle: shared start. Star: the minimum. Contour lines are evenly spaced in log(1 + L).', 'دایره توخالی: نقطه شروع مشترک. ستاره: کمینه. خطوط تراز با فاصله یکسان در log(1 + L) کشیده شده‌اند.'))}</p>
    </div>
  );
}

function UpdateDetail({ o }: { o: OptimizerName }) {
  const { tr } = useT();
  const [s] = useS();
  const { traces } = useTraces();
  const t = traces[o];
  const k = kOf(t, s.step);
  if (k >= t.grads.length)
    return <Callout tone={t.diverged ? 'err' : 'info'}>{t.diverged ? tr(bi('This run diverged: lower its learning rate.', 'این اجرا واگرا شد: نرخ یادگیری‌اش را کم کن.')) : tr(bi('Last step reached. Move the step slider back to inspect an update.', 'به آخرین قدم رسیدی. اسلایدر قدم را عقب ببر تا یک به‌روزرسانی را ببینی.'))}</Callout>;
  const g = t.grads[k];
  const before = t.states[k];
  const after = t.states[k + 1];
  const d = [t.points[k + 1][0] - t.points[k][0], t.points[k + 1][1] - t.points[k][1]];
  const lr = s.lr[o];
  const n = k + 1;
  return (
    <div className="stack" data-testid={`upd-${o}`}>
      <Tex display>{`\\theta_{${k}} = ${v2(t.points[k])},\\quad \\cg{g} = ${v2(g)}`}</Tex>
      {o === 'sgd' && <Tex display>{`\\Delta\\theta = -\\clr{${tex(lr, 4)}}\\cdot ${v2(g)} = ${v2(d, 4)}`}</Tex>}
      {o === 'momentum' && (
        <>
          <Tex display>{`v = ${tex(s.beta)}\\cdot ${v2(before.v)} + ${v2(g)} = ${v2(after.v)}`}</Tex>
          <Tex display>{`\\Delta\\theta = -\\clr{${tex(lr, 4)}}\\cdot v = ${v2(d, 4)}`}</Tex>
        </>
      )}
      {o === 'adam' && (
        <>
          <Tex display>{`m = ${v2(after.m)},\\quad s = ${v2(after.s)}`}</Tex>
          <Tex display>{`\\hat m = \\frac{m}{1-${tex(s.beta1)}^{${n}}} = ${v2(after.m.map((v) => v / (1 - s.beta1 ** n)))}`}</Tex>
          <Tex display>{`\\sqrt{\\hat s} = \\sqrt{\\frac{s}{1-${tex(s.beta2)}^{${n}}}} = ${v2(after.s.map((v) => Math.sqrt(v / (1 - s.beta2 ** n))))}`}</Tex>
          <Tex display>{`\\Delta\\theta = -\\clr{${tex(lr, 4)}}\\,\\frac{\\hat m}{\\sqrt{\\hat s}+\\varepsilon} = ${v2(d, 4)}`}</Tex>
        </>
      )}
      <table className="tbl">
        <thead>
          <tr>
            <th />
            <th className="num">w₁</th>
            <th className="num">w₂</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>|Δθ|</td>
            <td className="num">{fmt(Math.abs(d[0]), 4)}</td>
            <td className="num">{fmt(Math.abs(d[1]), 4)}</td>
          </tr>
          <tr>
            <td>{tr(bi('|Δθ| / |g| (effective step size)', '|Δθ| / |g| (اندازه قدم مؤثر)'))}</td>
            <td className="num c-lr">{fmt(Math.abs(d[0]) / Math.max(1e-12, Math.abs(g[0])), 4)}</td>
            <td className="num c-lr">{fmt(Math.abs(d[1]) / Math.max(1e-12, Math.abs(g[1])), 4)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function Right() {
  const { tr } = useT();
  const [s, set] = useS();
  return (
    <div className="stack">
      {s.obj === 'valley' ? (
        <>
          <Slider label="a" value={s.a} min={0.1} max={2} step={0.05} onChange={(a) => set({ a })} />
          <Slider label="b" value={s.b} min={1} max={20} step={0.5} onChange={(b) => set({ b })} testId="opt-b" />
          <P className="small muted">{bi(`SGD is stable only for $\\eta < 2/b = ${tex(2 / s.b, 4)}$.`, `SGD فقط برای $\\eta < 2/b = ${tex(2 / s.b, 4)}$ پایدار است.`)}</P>
        </>
      ) : (
        <Slider label="c" value={s.c} min={1} max={30} step={0.5} onChange={(c) => set({ c })} />
      )}
      <h4>{tr(bi('Learning rate per optimizer', 'نرخ یادگیری هر بهینه‌ساز'))}</h4>
      {OPTS.map((o) => (
        <Slider
          key={o}
          label={<span style={{ color: `var(${COLOR_VAR[o]})` }}>η {tr(NAME[o])}</span>}
          value={s.lr[o]}
          min={0.0005}
          max={o === 'adam' ? 1 : 0.4}
          step={0.0005}
          digits={4}
          onChange={(v) => set({ lr: { ...s.lr, [o]: v }, focus: o })}
          testId={`opt-lr-${o}`}
        />
      ))}
      <Slider label="β (momentum)" value={s.beta} min={0} max={0.99} step={0.01} onChange={(beta) => set({ beta, focus: 'momentum' })} />
      <Slider label="β₁ (Adam)" value={s.beta1} min={0} max={0.99} step={0.01} onChange={(beta1) => set({ beta1, focus: 'adam' })} />
      <Slider label="β₂ (Adam)" value={s.beta2} min={0.5} max={0.9999} step={0.0001} digits={4} onChange={(beta2) => set({ beta2, focus: 'adam' })} />
      <Slider label={tr(bi('steps', 'تعداد قدم'))} value={s.steps} min={10} max={500} step={10} digits={0} onChange={(steps) => set({ steps, step: Math.min(s.step, steps) })} />
      <h4>{tr(bi('One update, with numbers', 'یک به‌روزرسانی، با عدد'))}</h4>
      <Segmented<OptimizerName> label="optimizer" value={s.focus} onChange={(focus) => set({ focus })} options={OPTS.map((o) => ({ value: o, label: tr(NAME[o]) }))} />
      <UpdateDetail o={s.focus} />
    </div>
  );
}

function Bottom() {
  const { tr } = useT();
  const [s] = useS();
  const { obj, traces } = useTraces();
  const layers = useMemo<PlotLayer[]>(
    () => [
      ...OPTS.filter((o) => s.show[o]).map(
        (o): PlotLayer => ({ kind: 'polyline', pts: traces[o].losses.slice(0, kOf(traces[o], s.step) + 1).map((L, i) => [i, Math.log10(Math.max(L, 1e-10))]), color: `var(${COLOR_VAR[o]})`, width: 2 }),
      ),
      { kind: 'vline', at: s.step, color: 'var(--muted)' },
    ],
    [traces, s.step, s.show],
  );
  return (
    <div className="viz-grid">
      <div>
        <div className="small muted">{tr(bi('Loss vs step (log₁₀ scale)', 'خطا بر حسب قدم (مقیاس log₁₀)'))}</div>
        <Plot2D testId="opt-loss" xDomain={[0, s.steps]} yDomain={[-10, Math.max(1, Math.ceil(Math.log10(obj.f(obj.start))) + 1)]} layers={layers} height={210} xLabel={tr(bi('step k', 'قدم k'))} yLabel="log₁₀ L" />
      </div>
      <table className="tbl" data-testid="opt-table">
        <thead>
          <tr>
            <th>{tr(bi('optimizer', 'بهینه‌ساز'))}</th>
            <th className="num">η</th>
            <th className="num">{tr(bi(`L at step ${s.step}`, `L در قدم ${s.step}`))}</th>
            <th className="num">{tr(bi('distance to min', 'فاصله تا کمینه'))}</th>
          </tr>
        </thead>
        <tbody>
          {OPTS.map((o) => {
            const t = traces[o];
            const k = kOf(t, s.step);
            const p = t.points[k];
            return (
              <tr key={o}>
                <td style={{ color: `var(${COLOR_VAR[o]})`, fontWeight: 600 }}>{tr(NAME[o])}</td>
                <td className="num">{fmt(s.lr[o], 4)}</td>
                <td className="num c-L">{t.diverged && k === t.points.length - 1 ? tr(bi('diverged', 'واگرا')) : fmt(t.losses[k], 5)}</td>
                <td className="num">{fmt(Math.hypot(p[0] - obj.min[0], p[1] - obj.min[1]), 4)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export const lesson15: LessonDef = {
  id: ID,
  title: bi('Optimizers: SGD, Momentum, Adam', 'بهینه‌سازها: SGD، Momentum و Adam'),
  short: bi('Optimizers', 'بهینه‌سازها'),
  objectives: [
    bi('See why plain SGD zig-zags and crawls in a narrow valley', 'ببینی چرا SGD ساده در یک دره باریک زیگزاگ می‌زند و کند جلو می‌رود'),
    bi('Explain momentum as a velocity that accumulates gradients', 'Momentum را به شکل سرعتی که گرادیان‌ها را جمع می‌کند توضیح بدهی'),
    bi('Explain Adam as per-parameter adaptive step sizes', 'Adam را به شکل اندازه قدم تطبیقی برای هر پارامتر توضیح بدهی'),
    bi('Read the exact update equations with real numbers', 'فرمول‌های دقیق به‌روزرسانی را با عددهای واقعی بخوانی'),
  ],
  revealSteps: 5,
  Explain,
  Center,
  Right,
  Bottom,
  checkpoint: {
    question: bi('Which optimizer gives every parameter its own step size, based on that parameter\'s gradient history?', 'کدام بهینه‌ساز به هر پارامتر، بر اساس تاریخچه گرادیان همان پارامتر، اندازه قدم جداگانه‌ای می‌دهد؟'),
    choices: [
      { id: 'sgd', label: bi('SGD', 'SGD') },
      { id: 'momentum', label: bi('SGD with momentum', 'SGD با Momentum') },
      { id: 'adam', label: bi('Adam', 'Adam') },
    ],
    answer: 'adam',
    explain: bi(
      'Adam divides each parameter\'s averaged gradient by the root of its averaged squared gradient. Momentum smooths the direction but still uses one learning rate for every parameter. Compare the "effective step size" row for both in the right panel.',
      'Adam میانگین گرادیان هر پارامتر را بر جذر میانگین مربع گرادیان همان پارامتر تقسیم می‌کند. Momentum جهت را نرم می‌کند ولی هنوز برای همه پارامترها یک نرخ یادگیری دارد. ردیف «اندازه قدم مؤثر» را برای هر دو در پنل راست مقایسه کن.',
    ),
  },
};
