// Lesson 13: Learning rate. Four copies of the same one-weight problem (preset "lr-compare",
// L(w) = (2w − 10)², w0 = 3) are trained with real engine SGD steps, each with its own learning
// rate, and animated in sync with one shared step counter.

import { useEffect, useMemo } from 'react';
import type { LessonDef } from './types';
import { bi, useT, type Bi } from '../i18n';
import { useLab } from '../state/lab';
import { useLessonState } from '../state/lessonState';
import { Plot2D, type PlotLayer } from '../components/Plot2D';
import { Tex } from '../components/Tex';
import { Segmented, Slider, Stat } from '../components/controls';
import { Advanced, Callout, Cycle, P, Predict, Reveal } from '../components/teaching';
import { fmt, tex, texP } from '../engine';
import { curvatureAt, factor, lossAt, minimizerFrom, regime, runLr, stepsTo, type LrProblem, type LrRun, type Regime } from './math/l13';

const ID = 'l13';
const COLORS = ['var(--c-x)', 'var(--c-yhat)', 'var(--c-wneg)', 'var(--c-L)'];
const NAMES: Bi[] = [bi('Small', 'کوچک'), bi('Appropriate', 'مناسب'), bi('Large', 'بزرگ'), bi('Excessive', 'بیش از حد')];

interface S {
  lrs: number[];
  step: number;
  steps: number;
  playing: boolean;
  scale: 'linear' | 'log';
  focus: number;
}
const INIT: S = { lrs: [0.01, 0.1, 0.2, 0.27], step: 0, steps: 30, playing: false, scale: 'log', focus: 1 };
const useS = () => useLessonState<S>(ID, INIT);

const REGIME: Record<Regime, Bi> = {
  none: bi('no progress', 'بدون پیشرفت'),
  slow: bi('slow convergence', 'همگرایی کند'),
  fast: bi('fast convergence', 'همگرایی سریع'),
  oneStep: bi('lands in one step', 'در یک قدم می‌رسد'),
  oscillate: bi('overshoots, oscillates, converges', 'از کمینه رد می‌شود، نوسان می‌کند، همگرا می‌شود'),
  edge: bi('bounces forever', 'برای همیشه رفت‌وبرگشت می‌کند'),
  diverge: bi('diverges', 'واگرا می‌شود'),
};

/** Problem and runs, all from the lab store's network and data. */
function useRuns() {
  const [s] = useS();
  const net = useLab((st) => st.net);
  const data = useLab((st) => st.data);
  const loss = useLab((st) => st.loss);
  const lossOpts = useLab((st) => st.lossOpts);
  const frozen = useLab((st) => st.frozen);
  return useMemo(() => {
    const p: LrProblem = { net, data, loss, lossOpts, frozen };
    const w0 = net.layers[0].W[0][0];
    const curv = curvatureAt(p, w0);
    const wStar = minimizerFrom(p, w0);
    const runs = s.lrs.map((lr) => runLr(p, lr, s.steps));
    return { p, w0, curv, wStar, runs, x: data.X[0][0], y: data.Y[0][0] };
  }, [net, data, loss, lossOpts, frozen, s.lrs, s.steps]);
}

/** State of a run at the shared step (a diverged run stays at its last point). */
const at = (run: LrRun, k: number) => Math.min(k, run.w.length - 1);

function Explain() {
  const { tr } = useT();
  return (
    <>
      <Cycle active={2} />
      <Reveal at={1}>
        <h3>{tr(bi('1. The size of each step', '۱. اندازه هر قدم'))}</h3>
        <P>
          {bi(
            'Gradient descent decides the direction from the gradient. The learning rate $\\clr{\\eta}$ decides how far to go in that direction:',
            'گرادیان کاهشی جهت حرکت را از گرادیان می‌فهمد. نرخ یادگیری (learning rate) $\\clr{\\eta}$ تعیین می‌کند در آن جهت چقدر جلو برویم:',
          )}
        </P>
        <Tex display>{"\\cw{w_{k+1}} = \\cw{w_k} - \\clr{\\eta}\\,\\cg{L'(w_k)}"}</Tex>
        <P>
          {bi(
            'Think of walking downhill in fog: the slope under your feet tells you which way is down, and $\\eta$ is the length of your stride. Too short and you crawl. Too long and you jump over the valley.',
            'فرض کن در مه از تپه پایین می‌آیی: شیب زیر پایت می‌گوید پایین کدام طرف است و $\\eta$ طول گام توست. خیلی کوتاه باشد، فقط می‌خزی. خیلی بلند باشد، از روی دره می‌پری.',
          )}
        </P>
      </Reveal>
      <Reveal at={2}>
        <P>
          {bi(
            'The four panels train the same network from the same start $w_0 = 3$ on $L(w) = (2w-10)^2$. Only $\\eta$ differs. Press play and watch them in sync.',
            'چهار نمودار، یک شبکه را از یک نقطه شروع $w_0 = 3$ روی $L(w) = (2w-10)^2$ آموزش می‌دهند. فقط $\\eta$ فرق دارد. دکمه پخش را بزن و همزمان تماشا کن.',
          )}
        </P>
        <Predict
          id={ID + '-big'}
          question={bi('The 4th panel uses $\\eta = 0.27$, a bit more than the 3rd. What will it do?', 'نمودار چهارم $\\eta = 0.27$ دارد، کمی بیشتر از سومی. چه می‌کند؟')}
          choices={[
            { id: 'a', label: bi('Converge faster than the 3rd', 'سریع‌تر از سومی همگرا می‌شود') },
            { id: 'b', label: bi('Oscillate a little more, then settle', 'کمی بیشتر نوسان می‌کند و بعد آرام می‌گیرد') },
            { id: 'c', label: bi('Bounce farther away every step', 'هر قدم دورتر پرت می‌شود') },
          ]}
          answer="c"
          explain={bi(
            'Past $\\eta = 0.25$ every step overshoots by more than it corrects: $|w-5|$ grows by a factor $|1-8\\eta| = 1.16$ per step. A small change in $\\eta$ crossed a sharp threshold.',
            'بعد از $\\eta = 0.25$ هر قدم بیشتر از آنچه اصلاح می‌کند از هدف رد می‌شود: $|w-5|$ در هر قدم $|1-8\\eta| = 1.16$ برابر می‌شود. یک تغییر کوچک در $\\eta$ از یک آستانه تیز رد شد.',
          )}
        />
      </Reveal>
      <Reveal at={3}>
        <h3>{tr(bi('2. Four behaviours', '۲. چهار رفتار'))}</h3>
        <ul>
          <li>{tr(bi('Small η: always moves the right way, but needs many steps.', 'η کوچک: همیشه درست حرکت می‌کند، ولی قدم‌های زیادی لازم دارد.'))}</li>
          <li>{tr(bi('Appropriate η: a few steps and it is there.', 'η مناسب: با چند قدم می‌رسد.'))}</li>
          <li>{tr(bi('Large η: jumps past the minimum, then back, shrinking each time (oscillation).', 'η بزرگ: از کمینه رد می‌شود، برمی‌گردد و هر بار رفت‌وبرگشت کوچک‌تر می‌شود (نوسان).'))}</li>
          <li>{tr(bi('Excessive η: each jump is bigger than the last; the loss explodes (divergence).', 'η بیش از حد: هر پرش از قبلی بزرگ‌تر است؛ خطا منفجر می‌شود (واگرایی).'))}</li>
        </ul>
      </Reveal>
      <Reveal at={4}>
        <h3>{tr(bi('3. Where the threshold comes from', '۳. آستانه از کجا می‌آید'))}</h3>
        <P>
          {bi(
            "Here $L'(w) = 8(w-5)$, so one step gives $w_{k+1} - 5 = (1-8\\eta)(w_k - 5)$. The distance to the minimum is multiplied by the same factor $r = 1 - 8\\eta$ every step. The 8 is the curvature $L''$: how fast the slope changes.",
            "اینجا $L'(w) = 8(w-5)$ است، پس یک قدم می‌دهد $w_{k+1} - 5 = (1-8\\eta)(w_k - 5)$. فاصله تا کمینه در هر قدم در یک عدد ثابت $r = 1 - 8\\eta$ ضرب می‌شود. عدد ۸ همان انحنا (curvature) یعنی $L''$ است: اینکه شیب با چه سرعتی عوض می‌شود.",
          )}
        </P>
        <table className="tbl">
          <tbody>
            <tr><td><Tex>{'0 < \\eta < 0.125'}</Tex></td><td>{tr(bi('0 < r < 1: same side, shrinking', '0 < r < 1: همان طرف، کوچک‌شونده'))}</td></tr>
            <tr><td><Tex>{'\\eta = 0.125'}</Tex></td><td>{tr(bi('r = 0: lands on w = 5 in one step', 'r = 0: در یک قدم روی w = 5 می‌نشیند'))}</td></tr>
            <tr><td><Tex>{'0.125 < \\eta < 0.25'}</Tex></td><td>{tr(bi('−1 < r < 0: flips sides, still shrinking', '−1 < r < 0: طرفش عوض می‌شود، ولی کوچک می‌شود'))}</td></tr>
            <tr><td><Tex>{'\\eta > 0.25'}</Tex></td><td>{tr(bi('r < −1: flips and grows: diverges', 'r < −1: طرفش عوض می‌شود و بزرگ می‌شود: واگرا'))}</td></tr>
          </tbody>
        </table>
        <Callout>
          {tr(
            bi(
              'A real network has millions of directions, each with its own curvature, and the curvature changes as the weights move. There is no single magic threshold to compute, so the learning rate is tuned by experiment (and often lowered during training with a schedule).',
              'یک شبکه واقعی میلیون‌ها جهت دارد که هر کدام انحنای خودش را دارد، و انحنا با حرکت وزن‌ها عوض می‌شود. یک آستانه جادویی واحد وجود ندارد که حساب کنیم؛ برای همین نرخ یادگیری با آزمایش تنظیم می‌شود (و اغلب در طول آموزش با یک برنامه یا schedule کم می‌شود).',
            ),
          )}
        </Callout>
      </Reveal>
      <Reveal at={5}>
        <Advanced title={bi('Stability on a quadratic, and many dimensions', 'پایداری روی تابع درجه دو و چند بعد')}>
          <P>
            {bi(
              "Near any smooth minimum, $L(w) \\approx L(w^*) + \\tfrac12 L''(w^*)(w-w^*)^2$. Gradient descent then gives $w_{k} - w^* = (1-\\eta L'')^k (w_0 - w^*)$, which converges iff $|1-\\eta L''| < 1$, i.e. $0 < \\eta < 2/L''$. The fastest rate is $\\eta = 1/L''$ (Newton's step). At $\\eta = 2/L''$ exactly ($0.25$ here) it bounces between $3$ and $7$ forever.",
              "نزدیک هر کمینه نرم داریم $L(w) \\approx L(w^*) + \\tfrac12 L''(w^*)(w-w^*)^2$. گرادیان کاهشی می‌دهد $w_{k} - w^* = (1-\\eta L'')^k (w_0 - w^*)$ که فقط و فقط وقتی همگرا می‌شود که $|1-\\eta L''| < 1$، یعنی $0 < \\eta < 2/L''$. سریع‌ترین حالت $\\eta = 1/L''$ است (قدم نیوتن). دقیقاً در $\\eta = 2/L''$ (اینجا $0.25$) برای همیشه بین $3$ و $7$ رفت‌وبرگشت می‌کند.",
            )}
          </P>
          <P>
            {bi(
              'With many weights, the curvature is the Hessian matrix $H$. Along each eigenvector the same analysis holds with eigenvalue $\\lambda_i$, so stability needs $\\eta < 2/\\lambda_{\\max}$, while progress along the flattest direction is $1 - \\eta\\lambda_{\\min}$ per step. A large ratio $\\kappa = \\lambda_{\\max}/\\lambda_{\\min}$ forces a slow compromise. That is the problem momentum and Adam (next lesson) attack.',
              'با وزن‌های زیاد، انحنا ماتریس هسین $H$ است. در امتداد هر بردار ویژه همین تحلیل با مقدار ویژه $\\lambda_i$ برقرار است؛ پس پایداری $\\eta < 2/\\lambda_{\\max}$ لازم دارد، ولی پیشرفت در صاف‌ترین جهت در هر قدم فقط $1 - \\eta\\lambda_{\\min}$ است. نسبت بزرگ $\\kappa = \\lambda_{\\max}/\\lambda_{\\min}$ یک مصالحه کند را تحمیل می‌کند. این همان مشکلی است که Momentum و Adam (درس بعد) حل می‌کنند.',
            )}
          </P>
        </Advanced>
      </Reveal>
    </>
  );
}

function RunCard({ i }: { i: number }) {
  const { tr } = useT();
  const [s, set] = useS();
  const { p, curv, wStar, runs } = useRuns();
  const run = runs[i];
  const k = at(run, s.step);
  const reg = regime(run.lr, curv);
  const layers = useMemo<PlotLayer[]>(() => {
    const pts = run.w.slice(0, k + 1).map((w, j) => [w, run.loss[j]] as [number, number]);
    return [
      { kind: 'fn', f: (w) => lossAt(p, w), color: 'var(--line-strong)', width: 2 },
      { kind: 'vline', at: wStar, color: 'var(--muted)', dash: '3 4' },
      { kind: 'polyline', pts, color: COLORS[i], width: 1.6, markers: true },
      { kind: 'points', color: COLORS[i], pts: [{ x: run.w[k], y: run.loss[k], r: 6.5 }] },
    ];
  }, [p, run, k, wStar, i]);
  const gone = Math.abs(run.w[k] - wStar) > 6.5;
  return (
    <div className={`card ${s.focus === i ? '' : 'flat'}`} style={{ borderColor: s.focus === i ? COLORS[i] : undefined, padding: '10px 12px' }} onClick={() => set({ focus: i })} data-testid={`lr-card-${i}`}>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <strong style={{ color: COLORS[i] }}>
          {tr(NAMES[i])} · <Tex>{`\\eta = ${tex(run.lr, 4)}`}</Tex>
        </strong>
        <span className="small muted">{tr(REGIME[reg])}</span>
      </div>
      <Plot2D xDomain={[wStar - 6.5, wStar + 6.5]} yDomain={[0, 0.5 * curv * 6.5 ** 2 * 1.05]} layers={layers} height={170} xLabel="w" ariaLabel={`learning rate ${run.lr}`} />
      <div className="small row" style={{ justifyContent: 'space-between' }}>
        <span className="num">
          k = {k} · w = {fmt(run.w[k], 4)} · L = <span className="c-L">{fmt(run.loss[k], 4)}</span>
        </span>
        {(run.diverged && s.step >= run.w.length - 1) || gone ? <span className="c-neg">{tr(bi('left the plot', 'از نمودار بیرون رفت'))}</span> : null}
      </div>
    </div>
  );
}

function Center() {
  const { tr } = useT();
  const [s, set] = useS();
  // Shared clock: advances one step at a time for all four runs.
  useEffect(() => {
    if (!s.playing) return;
    const id = window.setInterval(() => set((q) => (q.step >= q.steps ? { playing: false } : { step: q.step + 1 })), 380);
    return () => window.clearInterval(id);
  }, [s.playing, set]);
  return (
    <div className="viz">
      <div className="row">
        <button className={`btn ${s.playing ? '' : 'primary'}`} onClick={() => set({ playing: !s.playing, step: !s.playing && s.step >= s.steps ? 0 : s.step })} data-testid="lr-play">
          {s.playing ? `❚❚ ${tr(bi('Pause', 'توقف'))}` : `▶ ${tr(bi('Play', 'پخش'))}`}
        </button>
        <button className="btn" onClick={() => set({ step: Math.min(s.steps, s.step + 1), playing: false })} data-testid="lr-next">
          +1 {tr(bi('step', 'قدم'))}
        </button>
        <button className="btn ghost" onClick={() => set({ step: 0, playing: false })}>
          ↺
        </button>
        <div className="grow">
          <Slider label={tr(bi('step k', 'قدم k'))} value={s.step} min={0} max={s.steps} step={1} digits={0} onChange={(step) => set({ step, playing: false })} testId="lr-step" />
        </div>
      </div>
      <div className="viz-grid">
        {[0, 1, 2, 3].map((i) => (
          <RunCard key={i} i={i} />
        ))}
      </div>
      <p className="small muted">
        {tr(bi('Grey curve: the loss L(w). Dashed line: the minimum. Dots: the weights visited, joined in order. Click a panel to inspect its numbers.', 'منحنی خاکستری: خطا L(w). خط‌چین: کمینه. نقطه‌ها: وزن‌هایی که به ترتیب دیده شده‌اند. روی هر نمودار کلیک کن تا عددهایش را ببینی.'))}
      </p>
    </div>
  );
}

function Right() {
  const { tr } = useT();
  const [s, set] = useS();
  const { curv, wStar, runs, w0, x, y } = useRuns();
  const setParam = useLab((st) => st.setParam);
  const run = runs[s.focus];
  const k = at(run, s.step);
  const canStep = k < run.w.length - 1;
  return (
    <div className="stack">
      <Tex display>{`\\cL{L(w)} = (\\cx{${tex(x)}}\\,\\cw{w} - \\cy{${tex(y)}})^2`}</Tex>
      <div className="stat-grid">
        <Stat k={<Tex>{"L''(w)"}</Tex>} v={fmt(curv, 4)} testId="curvature" />
        <Stat k={<Tex>{'w^*'}</Tex>} v={fmt(wStar, 4)} />
        <Stat k={<Tex>{"1/L''"}</Tex>} v={fmt(1 / curv, 4)} color="lr" />
        <Stat k={<Tex>{"2/L''"}</Tex>} v={fmt(2 / curv, 4)} color="lr" testId="lr-max" />
      </div>
      <Slider label={tr(bi('start w₀ (all runs)', 'شروع w₀ (همه)'))} value={w0} min={-1} max={11} step={0.1} color="w" onChange={(v) => setParam({ kind: 'W', layer: 0, j: 0, i: 0 }, v)} testId="lr-w0" />
      <h4>{tr(bi('Learning rates', 'نرخ‌های یادگیری'))}</h4>
      {runs.map((r, i) => (
        <div key={i}>
          <Slider
            label={<span style={{ color: COLORS[i] }}>η {i + 1}</span>}
            value={s.lrs[i]}
            min={0.001}
            max={0.35}
            step={0.001}
            digits={4}
            onChange={(v) => set({ lrs: s.lrs.map((o, j) => (j === i ? v : o)), focus: i })}
            testId={`lr-${i}`}
          />
          <div className="small muted num" style={{ marginInlineStart: '5.5em' }}>
            r = 1 − η·L″ = {fmt(factor(r.lr, curv), 4)} · {tr(REGIME[regime(r.lr, curv)])}
          </div>
        </div>
      ))}
      <Slider label={tr(bi('steps', 'تعداد قدم'))} value={s.steps} min={5} max={100} step={1} digits={0} onChange={(steps) => set({ steps, step: Math.min(s.step, steps) })} />
      <h4 style={{ color: COLORS[s.focus] }}>
        {tr(NAMES[s.focus])}: {tr(bi('step', 'قدم'))} {k} → {k + 1}
      </h4>
      {canStep ? (
        <>
          <Tex display>{`\\cg{L'(w_{${k}})} = 2\\cdot${texP(x)}\\,(${tex(x)}\\cdot${texP(run.w[k], 4)} - ${texP(y)}) = \\cg{${tex(run.grad[k], 4)}}`}</Tex>
          <Tex display>{`\\cw{w_{${k + 1}}} = ${tex(run.w[k], 4)} - \\clr{${tex(run.lr, 4)}}\\cdot${texP(run.grad[k], 4)} = \\cw{${tex(run.w[k + 1], 4)}}`}</Tex>
          <Tex display>{`\\cL{L(w_{${k + 1}})} = \\cL{${tex(run.loss[k + 1], 4)}}`}</Tex>
        </>
      ) : (
        <Callout tone={run.diverged ? 'err' : 'info'}>
          {run.diverged
            ? tr(bi(`The loss passed 10⁹ after ${run.w.length - 1} steps: training was stopped (diverged).`, `خطا بعد از ${run.w.length - 1} قدم از ۱۰⁹ گذشت: آموزش متوقف شد (واگرایی).`))
            : tr(bi('Last step reached. Increase the number of steps or press ↺.', 'به آخرین قدم رسیدی. تعداد قدم‌ها را بیشتر کن یا ↺ را بزن.'))}
        </Callout>
      )}
      <div className="stat-grid">
        <Stat k="|w − w*|" v={fmt(Math.abs(run.w[k] - wStar), 5)} />
        <Stat k={tr(bi('steps to L < 0.01', 'قدم تا L < 0.01'))} v={stepsTo(run, 0.01) < 0 ? '—' : stepsTo(run, 0.01)} />
      </div>
    </div>
  );
}

function Bottom() {
  const { tr } = useT();
  const [s, set] = useS();
  const { runs, curv } = useRuns();
  const log = s.scale === 'log';
  const lossLayers = useMemo<PlotLayer[]>(() => {
    const yv = (L: number) => (log ? Math.log10(Math.max(L, 1e-8)) : L);
    return [
      ...runs.map((r, i): PlotLayer => ({ kind: 'polyline', pts: r.loss.slice(0, at(r, s.step) + 1).map((L, j) => [j, yv(L)]), color: COLORS[i], width: 2, markers: s.steps <= 40 })),
      { kind: 'vline', at: s.step, color: 'var(--muted)' },
    ];
  }, [runs, s.step, s.steps, log]);
  const L0 = runs[0].loss[0];
  const lrMax = Math.max(0.35, ...s.lrs);
  const rLayers: PlotLayer[] = [
    { kind: 'band', x0: 0, x1: 2 / curv, color: 'var(--c-yhat)', opacity: 0.08, label: tr(bi('converges', 'همگرا')) },
    { kind: 'band', x0: 2 / curv, x1: lrMax, color: 'var(--c-L)', opacity: 0.08, label: tr(bi('diverges', 'واگرا')) },
    { kind: 'hline', at: 1, color: 'var(--c-L)', label: '|r| = 1' },
    { kind: 'fn', f: (lr) => Math.abs(factor(lr, curv)), color: 'lr', width: 2.2 },
    { kind: 'points', color: 'lr', pts: s.lrs.map((lr, i) => ({ x: lr, y: Math.abs(factor(lr, curv)), color: COLORS[i] })) },
  ];
  return (
    <div className="viz-grid">
      <div>
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <span className="small muted">{tr(bi('Loss vs step (all runs, same clock)', 'خطا بر حسب قدم (همه اجراها، یک ساعت مشترک)'))}</span>
          <Segmented
            label="scale"
            value={s.scale}
            onChange={(scale) => set({ scale })}
            options={[
              { value: 'linear', label: tr(bi('linear', 'خطی')) },
              { value: 'log', label: 'log₁₀' },
            ]}
          />
        </div>
        <Plot2D testId="lr-loss-steps" xDomain={[0, s.steps]} yDomain={log ? [-8, Math.max(2, Math.ceil(Math.log10(L0)) + 4)] : [0, Math.max(1, L0 * 1.6)]} layers={lossLayers} height={210} xLabel={tr(bi('step k', 'قدم k'))} yLabel={log ? 'log₁₀ L' : 'L'} />
      </div>
      <div>
        <div className="small muted">{tr(bi('Factor |r| = |1 − η·L″| applied to the distance each step', 'ضریب |r| = |1 − η·L″| که هر قدم در فاصله ضرب می‌شود'))}</div>
        <Plot2D xDomain={[0, lrMax]} yDomain={[0, 2]} layers={rLayers} height={230} xLabel="η" yLabel="|r|" />
      </div>
    </div>
  );
}

export const lesson13: LessonDef = {
  id: ID,
  title: bi('Learning Rate', 'نرخ یادگیری'),
  short: bi('Learning rate', 'نرخ یادگیری'),
  objectives: [
    bi('Explain the learning rate as the size of each gradient step', 'نرخ یادگیری را به عنوان اندازه هر قدم گرادیان توضیح بدهی'),
    bi('Recognize slow convergence, fast convergence, oscillation and divergence', 'همگرایی کند، همگرایی سریع، نوسان و واگرایی را تشخیص بدهی'),
    bi("Derive the threshold η < 2/L″ = 0.25 for L(w) = (2w − 10)²", "آستانه η < 2/L″ = 0.25 را برای L(w) = (2w − 10)² به دست بیاوری"),
    bi('Explain why real networks need the learning rate tuned by experiment', 'توضیح بدهی چرا در شبکه‌های واقعی نرخ یادگیری با آزمایش تنظیم می‌شود'),
  ],
  presetId: 'lr-compare',
  revealSteps: 5,
  Explain,
  Center,
  Right,
  Bottom,
  checkpoint: {
    question: bi('On $L(w) = (2w-10)^2$, which learning rate reaches the minimum in exactly one step?', 'روی $L(w) = (2w-10)^2$ کدام نرخ یادگیری دقیقاً در یک قدم به کمینه می‌رسد؟'),
    choices: [
      { id: 'a', label: bi('$\\eta = 0.05$', '$\\eta = 0.05$') },
      { id: 'b', label: bi('$\\eta = 0.125$', '$\\eta = 0.125$') },
      { id: 'c', label: bi('$\\eta = 0.25$', '$\\eta = 0.25$') },
    ],
    answer: 'b',
    explain: bi(
      'The factor is $r = 1 - 8\\eta$; it is 0 at $\\eta = 1/8 = 0.125$. At $0.25$ the factor is $-1$ and the weight bounces between 3 and 7 forever. Set a slider to 0.125 to check.',
      'ضریب $r = 1 - 8\\eta$ است و در $\\eta = 1/8 = 0.125$ صفر می‌شود. در $0.25$ ضریب $-1$ است و وزن برای همیشه بین ۳ و ۷ رفت‌وبرگشت می‌کند. یکی از اسلایدرها را روی 0.125 بگذار و ببین.',
    ),
  },
};
