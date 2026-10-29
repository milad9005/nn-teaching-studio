// Lesson 12: gradient descent on L(w) = (2w − 10)².
// Shared lab store (preset "gd-step": w = 3, lr = 0.05, b frozen). Every update is the engine's
// optimizerStep; the trajectory is history[].theta; the update equation uses lab.lastUpdate.

import { useEffect, useMemo } from 'react';
import type { LessonDef } from './types';
import { bi, useT } from '../i18n';
import { useLab, useLabDerived } from '../state/lab';
import { useUi } from '../state/ui';
import { useLessonState } from '../state/lessonState';
import { Plot2D, type PlotLayer } from '../components/Plot2D';
import { LossHistory } from '../components/LabPanels';
import { Tex } from '../components/Tex';
import { Slider, Stat } from '../components/controls';
import { Advanced, Callout, Cycle, P, Predict, Reveal } from '../components/teaching';
import { batchLossGrad, clamp, datasetLoss, fmt, paramRefs, setParam, tex, texP } from '../engine';
import { W_REF } from './math/l08';
import { curvature, gdFactor, wStar } from './math/l12';

const ID = 'l12';

interface S {
  /** Slow motion: `speed` updates per second (lesson timer) instead of `speed` updates per frame (lab loop). */
  slow: boolean;
  playing: boolean;
}
const INIT: S = { slow: true, playing: false };
const useS = () => useLessonState<S>(ID, INIT);

const WD: [number, number] = [0, 10];
const LD: [number, number] = [-10, 110];
const r2 = (v: number) => Math.round(v * 100) / 100;

/**
 * Start a fresh run at weight w: reload the preset (fresh schedule, history and optimizer state),
 * keep the chosen learning rate, then set the weight without keeping the old trajectory.
 */
function startFrom(w: number) {
  const st = useLab.getState();
  const lr = st.opt.lr;
  st.setRunning(false);
  st.resetTraining();
  if (useLab.getState().opt.lr !== lr) useLab.getState().setOpt({ lr });
  const s2 = useLab.getState();
  s2.setNet(setParam(s2.net, W_REF, w), false);
}

function useGD() {
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const loss = useLab((s) => s.loss);
  const opt = useLab((s) => s.opt);
  const history = useLab((s) => s.history);
  const iteration = useLab((s) => s.train.schedule.iteration);
  const running = useLab((s) => s.running);
  const last = useLab((s) => s.lastUpdate);
  const error = useLab((s) => s.error);
  const { full } = useLabDerived();
  const x = data.X[0][0];
  const y = data.Y[0][0];
  const b = net.layers[0].b[0];
  const w = net.layers[0].W[0][0];
  const wIdx = paramRefs(net).findIndex((p) => p.id === W_REF.id);
  const traj = history.filter((h) => h.theta).map((h) => [h.theta![wIdx], h.loss] as [number, number]);
  const fL = (v: number) => datasetLoss(setParam(net, W_REF, v), data, loss);
  const gradAt = (v: number) => batchLossGrad(setParam(net, W_REF, v), data, loss).grads.W[0][0][0];
  return { net, data, loss, opt, history, iteration, running, last, error, x, y, b, w, L: full.loss, g: full.grads.W[0][0][0], lr: opt.lr, traj, fL, gradAt, ws: wStar(x, y, b) };
}

function Explain() {
  const { tr } = useT();
  return (
    <>
      <Cycle active={2} />
      <Reveal at={1}>
        <h3>{tr(bi('1. Walking downhill in fog', '۱. پایین رفتن از تپه در مه'))}</h3>
        <P>
          {bi(
            'Training wants the weight with the lowest loss. But the network never sees the whole loss curve: at the current $\\cw{w}$ it can only compute the loss and its slope $\\cg{dL/dw}$. Like standing on a hill in fog, you only feel which way the ground tilts under your feet.',
            'آموزش دنبال وزنی با کمترین خطاست. اما شبکه هیچ‌وقت کل منحنی خطا را نمی‌بیند: در $\\cw{w}$ فعلی فقط می‌تواند خطا و شیبش $\\cg{dL/dw}$ را حساب کند. مثل ایستادن روی تپه‌ای در مه: فقط حس می‌کنی زمین زیر پایت به کدام طرف کج است.',
          )}
        </P>
      </Reveal>
      <Reveal at={2}>
        <h3>{tr(bi('2. Go against the gradient', '۲. خلاف جهت گرادیان برو'))}</h3>
        <P>
          {bi(
            'The derivative points uphill: a positive slope means the loss rises to the right, so we should move left; a negative slope means it rises to the left, so we move right. Either way: step in the direction of $-\\cg{dL/dw}$, the negative gradient.',
            'مشتق به سمت سربالایی اشاره می‌کند: شیب مثبت یعنی خطا به سمت راست بالا می‌رود، پس باید به چپ برویم؛ شیب منفی یعنی به سمت چپ بالا می‌رود، پس به راست می‌رویم. در هر دو حالت: در جهت $-\\cg{dL/dw}$، یعنی منفی گرادیان، قدم بردار.',
          )}
        </P>
        <Predict
          id={ID + '-dir'}
          question={bi('At $w = 3$ the slope is $dL/dw = -16$. Which way should $w$ move to lower the loss?', 'در $w = 3$ شیب $dL/dw = -16$ است. $w$ باید به کدام طرف برود تا خطا کم شود؟')}
          choices={[
            { id: 'left', label: bi('Left (smaller w)', 'چپ (w کوچک‌تر)') },
            { id: 'right', label: bi('Right (larger w)', 'راست (w بزرگ‌تر)') },
            { id: 'stay', label: bi('Stay: it is already the minimum', 'بماند: همین حالا کمینه است') },
          ]}
          answer="right"
          explain={bi(
            'Negative slope: the loss goes down to the right. $-dL/dw = +16$ points right. Press "Step" and watch $w$ go from 3 to 3.8.',
            'شیب منفی: خطا به سمت راست پایین می‌آید. $-dL/dw = +16$ به راست اشاره می‌کند. «یک گام» را بزن و ببین $w$ از ۳ به ۳٫۸ می‌رود.',
          )}
        />
      </Reveal>
      <Reveal at={3}>
        <h3>{tr(bi('3. The update rule', '۳. قانون به‌روزرسانی'))}</h3>
        <div className="formula-hero">
          <Tex display>{'\\cw{w_{\\text{new}}} = \\cw{w_{\\text{old}}} - \\clr{\\eta}\\cdot\\cg{\\frac{dL}{dw}}'}</Tex>
        </div>
        <P>
          {bi(
            'The learning rate $\\clr{\\eta}$ scales the step. With $\\eta = 0.05$: $w_{\\text{new}} = 3 - 0.05\\cdot(-16) = 3.8$, and the loss drops from $16$ to $(7.6-10)^2 = 5.76$. The minus sign is what turns "uphill" into "downhill".',
            'نرخ یادگیری (learning rate) $\\clr{\\eta}$ اندازه قدم را تعیین می‌کند. با $\\eta = 0.05$: $w_{\\text{new}} = 3 - 0.05\\cdot(-16) = 3.8$ و خطا از $16$ به $(7.6-10)^2 = 5.76$ می‌رسد. همین علامت منفی است که «سربالایی» را به «سرازیری» تبدیل می‌کند.',
          )}
        </P>
      </Reveal>
      <Reveal at={4}>
        <Predict
          id={ID + '-size'}
          question={bi('With a fixed $\\eta$, as $w$ gets close to the minimum $w = 5$, the steps…', 'با $\\eta$ ثابت، وقتی $w$ به کمینه $w = 5$ نزدیک می‌شود، قدم‌ها…')}
          choices={[
            { id: 'bigger', label: bi('get bigger', 'بزرگ‌تر می‌شوند') },
            { id: 'smaller', label: bi('get smaller', 'کوچک‌تر می‌شوند') },
            { id: 'same', label: bi('stay the same size', 'همان اندازه می‌مانند') },
          ]}
          answer="smaller"
          explain={bi(
            'The step is $\\eta\\cdot|dL/dw|$, and the slope flattens near the bottom. Press "Run" and watch the dots on the curve bunch up near $w = 5$.',
            'اندازه قدم $\\eta\\cdot|dL/dw|$ است و شیب نزدیک ته کاسه صاف می‌شود. «اجرا» را بزن و ببین نقطه‌ها روی منحنی نزدیک $w = 5$ فشرده می‌شوند.',
          )}
        />
      </Reveal>
      <Reveal at={5}>
        <h3>{tr(bi('4. Repeat until flat', '۴. تکرار تا صاف شدن'))}</h3>
        <P>
          {bi(
            'Gradient descent is just this rule in a loop: compute the gradient, take a step, repeat. It stops making progress where the slope is zero. Drag the point anywhere on the curve to start a new run there, and change $\\eta$ to see slower, faster, or overshooting runs (the next lesson studies this).',
            'گرادیان کاهشی (gradient descent) همین قانون در یک حلقه است: گرادیان را حساب کن، یک قدم بردار، تکرار کن. جایی که شیب صفر است دیگر پیشرفتی نمی‌کند. نقطه را هر جای منحنی بکش تا اجرای تازه‌ای از آنجا شروع شود و $\\eta$ را تغییر بده تا اجراهای کندتر، سریع‌تر یا جهنده را ببینی (درس بعد همین را بررسی می‌کند).',
          )}
        </P>
        <Advanced title={bi('Why a small step always helps, and this curve exactly', 'چرا یک قدم کوچک همیشه کمک می‌کند، و این منحنی به‌طور دقیق')}>
          <P>
            {bi(
              'Locally $L(w + \\Delta) \\approx L(w) + L\'(w)\\,\\Delta$. With $\\Delta = -\\eta L\'(w)$ this gives $L(w) - \\eta\\,L\'(w)^2$, which is lower whenever $L\'(w) \\neq 0$ and $\\eta$ is small enough for the linear approximation to hold.',
              'به‌صورت محلی $L(w + \\Delta) \\approx L(w) + L\'(w)\\,\\Delta$. با $\\Delta = -\\eta L\'(w)$ این می‌شود $L(w) - \\eta\\,L\'(w)^2$ که هر وقت $L\'(w) \\neq 0$ باشد و $\\eta$ آن‌قدر کوچک باشد که تقریب خطی معتبر بماند، کمتر است.',
            )}
          </P>
          <P>
            {bi(
              'For $L(w) = (xw + b - y)^2$ the slope is linear in $w$ with curvature $L\'\' = 2x^2$, so every step multiplies the distance to $w^*$ by the same factor: $w_{\\text{new}} - w^* = (1 - \\eta\\cdot 2x^2)(w_{\\text{old}} - w^*)$. It converges when $|1 - 2\\eta x^2| < 1$, i.e. $0 < \\eta < 1/x^2$, and $\\eta = 1/(2x^2)$ lands on $w^*$ in one step. The live factor is shown on the right.',
              'برای $L(w) = (xw + b - y)^2$ شیب نسبت به $w$ خطی است و انحنایش $L\'\' = 2x^2$ است، پس هر قدم فاصله تا $w^*$ را در یک ضریب ثابت ضرب می‌کند: $w_{\\text{new}} - w^* = (1 - \\eta\\cdot 2x^2)(w_{\\text{old}} - w^*)$. وقتی همگرا می‌شود که $|1 - 2\\eta x^2| < 1$ باشد، یعنی $0 < \\eta < 1/x^2$، و $\\eta = 1/(2x^2)$ در یک قدم روی $w^*$ می‌نشیند. ضریب زنده در سمت راست نشان داده شده است.',
            )}
          </P>
        </Advanced>
      </Reveal>
    </>
  );
}

/** Play/pause/step/restart, shared by the center panel. Slow mode runs its own timer of engine steps. */
function Controls() {
  const { t, tr } = useT();
  const [s, set] = useS();
  const c = useGD();
  const speed = useUi((u) => u.speed);
  const setUi = useUi((u) => u.set);
  const optimizerStep = useLab((st) => st.optimizerStep);
  const setRunning = useLab((st) => st.setRunning);
  const active = s.playing || c.running;

  useEffect(() => {
    if (!s.playing) return;
    const id = setInterval(() => {
      const st = useLab.getState();
      const lastH = st.history[st.history.length - 1];
      if (st.error || (lastH && lastH.loss < 1e-10) || st.train.schedule.iteration >= 2000 || !st.optimizerStep()) set({ playing: false });
    }, 1000 / Math.max(1, speed));
    return () => clearInterval(id);
  }, [s.playing, speed, set]);
  // Never keep a lesson timer alive after leaving the lesson.
  useEffect(() => () => set({ playing: false }), [set]);

  const play = () => (s.slow ? set({ playing: true }) : setRunning(true));
  const pause = () => {
    set({ playing: false });
    setRunning(false);
  };
  const restart = () => {
    pause();
    startFrom(c.traj[0]?.[0] ?? c.w);
  };
  return (
    <div className="stack">
      <div className="row">
        <button className="btn" onClick={() => optimizerStep()} disabled={active} data-testid="l12-step">
          ⟳ {tr(bi('Step', 'یک گام'))}
        </button>
        <button className={`btn ${active ? '' : 'primary'}`} onClick={active ? pause : play} data-testid="l12-run">
          {active ? `❚❚ ${t('run.pause')}` : `▶ ${tr(bi('Run', 'اجرا'))}`}
        </button>
        <button className="btn" onClick={restart} data-testid="l12-reset">
          ↺ {tr(bi('Restart run', 'شروع دوباره'))}
        </button>
        <label className="small muted row" style={{ gap: 6 }}>
          {t('run.speed')}
          <input type="range" min={1} max={30} value={speed} onChange={(e) => setUi({ speed: Number(e.target.value) })} aria-label={t('run.speed')} />
          <span className="num">
            {speed} {s.slow ? tr(bi('steps/s', 'گام/ثانیه')) : t('run.stepsPerFrame')}
          </span>
        </label>
        <label className="small row" style={{ gap: 4 }}>
          <input type="checkbox" checked={s.slow} onChange={(e) => (pause(), set({ slow: e.target.checked }))} data-testid="l12-slow" />
          {tr(bi('slow motion', 'حرکت آهسته'))}
        </label>
      </div>
      {c.error && <div className="callout err small">{t(`err.${c.error}`)}</div>}
    </div>
  );
}

function Center() {
  const { tr } = useT();
  const c = useGD();
  const [, set] = useS();
  const step = c.lr * c.g;
  const layers = useMemo<PlotLayer[]>(() => {
    const out: PlotLayer[] = [];
    if (Number.isFinite(c.ws)) out.push({ kind: 'vline', at: c.ws, color: 'var(--line-strong)', label: `w* = ${fmt(c.ws, 3)}` });
    out.push(
      { kind: 'fn', f: c.fL, color: 'L', width: 2.8 },
      { kind: 'polyline', pts: c.traj, color: 'w', width: 1.6, markers: true, opacity: 0.85 },
      { kind: 'line', p: [c.w, c.L], slope: c.g, color: 'g', width: 1.8, dash: '7 5' },
    );
    if (c.last && c.traj.length > 1) {
      const a = c.traj[c.traj.length - 2];
      const b = c.traj[c.traj.length - 1];
      out.push({ kind: 'segment', from: a, to: b, color: 'lr', width: 2.4, arrow: true });
    }
    // Along the w-axis: the scaled gradient (uphill) and the actual step −η·dL/dw (downhill), mirror images.
    const yA = -5;
    if (Math.abs(step) > 1e-6)
      out.push(
        { kind: 'segment', from: [c.w, yA], to: [c.w + step, yA], color: 'g', width: 2, arrow: true, label: '+η·dL/dw' },
        { kind: 'segment', from: [c.w, yA], to: [c.w - step, yA], color: 'lr', width: 2.6, arrow: true, label: '−η·dL/dw' },
      );
    out.push({ kind: 'points', color: 'L', pts: [{ x: c.w, y: c.L, r: 0.1, label: `L = ${fmt(c.L, 4)}` }] });
    return out;
  }, [c.ws, c.fL, c.traj, c.w, c.L, c.g, c.last, step]);
  return (
    <div className="viz">
      <Controls />
      <div className="legend-chips">
        <span className="chip">
          <i style={{ background: 'var(--c-L)' }} /> {tr(bi('loss L(w)', 'خطا L(w)'))}
        </span>
        <span className="chip">
          <i style={{ background: 'var(--c-w)' }} /> {tr(bi('visited weights', 'وزن‌های طی‌شده'))}
        </span>
        <span className="chip">
          <i style={{ background: 'var(--c-g)' }} /> {tr(bi('tangent / gradient (uphill)', 'مماس / گرادیان (سربالایی)'))}
        </span>
        <span className="chip">
          <i style={{ background: 'var(--c-lr)' }} /> {tr(bi('step −η·dL/dw (downhill)', 'قدم −η·dL/dw (سرازیری)'))}
        </span>
      </div>
      <Plot2D
        testId="l12-curve"
        xDomain={WD}
        yDomain={LD}
        layers={layers}
        height={380}
        xLabel={tr(bi('weight w', 'وزن w'))}
        yLabel={tr(bi('loss L(w)', 'خطا L(w)'))}
        handle={{ x: c.w, y: c.L, color: 'L' }}
        onDrag={(v) => {
          set({ playing: false });
          startFrom(r2(clamp(v, WD[0], WD[1])));
        }}
        ariaLabel="gradient descent on the loss curve"
      />
      <div className="stat-grid" data-testid="l12-stats">
        <Stat k={tr(bi('step', 'گام'))} v={c.iteration} testId="l12-iter" />
        <Stat k={tr(bi('weight w', 'وزن w'))} v={fmt(c.w, 5)} color="w" testId="l12-w" />
        <Stat k={tr(bi('loss L', 'خطا L'))} v={fmt(c.L, 5)} color="L" testId="l12-loss" />
        <Stat k="dL/dw" v={fmt(c.g, 5)} color="g" testId="l12-grad" />
        <Stat k="η" v={fmt(c.lr, 4)} color="lr" />
      </div>
      <p className="small muted">{tr(bi('Drag the point to start a new run from any weight.', 'نقطه را بکش تا اجرای تازه‌ای از هر وزنی شروع شود.'))}</p>
    </div>
  );
}

function Right() {
  const { tr } = useT();
  const c = useGD();
  const [, set] = useS();
  const setOpt = useLab((st) => st.setOpt);
  const last = c.last;
  const before = last ? last.before.layers[0].W[0][0] : NaN;
  const gLast = last ? last.grads.W[0][0][0] : NaN;
  const wIdx = paramRefs(c.net).findIndex((p) => p.id === W_REF.id);
  const after = last ? before + last.delta[wIdx] : NaN;
  const Lbefore = c.traj.length > 1 ? c.traj[c.traj.length - 2][1] : NaN;
  const factor = gdFactor(c.lr, c.x);
  const sign = c.g > 1e-9 ? 'pos' : c.g < -1e-9 ? 'neg' : 'zero';
  return (
    <div className="stack">
      <Slider label={tr(bi('learning rate η', 'نرخ یادگیری η'))} value={c.lr} min={0.001} max={0.3} step={0.001} color="lr" digits={4} onChange={(lr) => setOpt({ lr })} testId="l12-lr" />
      <Slider
        label={tr(bi('start weight w', 'وزن شروع w'))}
        value={c.w}
        min={WD[0]}
        max={WD[1]}
        step={0.01}
        color="w"
        onChange={(v) => {
          set({ playing: false });
          startFrom(v);
        }}
        testId="l12-w-slider"
      />

      <h4>{tr(bi('Last update', 'آخرین به‌روزرسانی'))}</h4>
      {last ? (
        <>
          <Tex display>{`\\cw{w_{\\text{new}}} = \\cw{w_{\\text{old}}} - \\clr{\\eta}\\cdot\\cg{\\tfrac{dL}{dw}} = ${tex(before, 4)} - ${tex(last.lr, 4)}\\cdot${texP(gLast, 4)} = \\cw{${tex(after, 4)}}`}</Tex>
          <Tex display>{`\\cL{L}:\\ ${tex(Lbefore, 4)} \\;\\to\\; ${tex(c.traj[c.traj.length - 1][1], 4)}`}</Tex>
        </>
      ) : (
        <P className="small muted">{bi('No update yet in this run. Press "Step".', 'هنوز در این اجرا به‌روزرسانی‌ای نشده است. «یک گام» را بزن.')}</P>
      )}

      <h4>{tr(bi('Next update (preview)', 'به‌روزرسانی بعدی (پیش‌نمایش)'))}</h4>
      <Tex display>{`\\cg{\\tfrac{dL}{dw}} = 2(\\cw{w}\\cx{x} + \\cb{b} - \\cy{y})\\cdot\\cx{x} = 2(${texP(c.w, 4)}\\cdot${texP(c.x)} + ${texP(c.b)} - ${texP(c.y)})\\cdot${texP(c.x)} = ${tex(c.g, 4)}`}</Tex>
      <Tex display>{`\\cw{w} - \\clr{\\eta}\\cdot\\cg{\\tfrac{dL}{dw}} = ${tex(c.w, 4)} - ${tex(c.lr, 4)}\\cdot${texP(c.g, 4)} = \\cw{${tex(c.w - c.lr * c.g, 4)}}`}</Tex>
      <Callout tone={sign === 'zero' ? 'info' : 'warn'}>
        {sign === 'pos' && tr(bi('Slope positive → the loss rises to the right → the update moves w LEFT.', 'شیب مثبت ← خطا به راست بالا می‌رود ← به‌روزرسانی w را به چپ می‌برد.'))}
        {sign === 'neg' && tr(bi('Slope negative → the loss rises to the left → the update moves w RIGHT.', 'شیب منفی ← خطا به چپ بالا می‌رود ← به‌روزرسانی w را به راست می‌برد.'))}
        {sign === 'zero' && tr(bi('Slope zero → no update: we are at the bottom.', 'شیب صفر ← به‌روزرسانی‌ای نیست: در ته کاسه هستیم.'))}
      </Callout>
      <div className="stat-grid">
        <Stat k="L″ = 2x²" v={fmt(curvature(c.x), 4)} />
        <Stat k={tr(bi('factor 1 − η·L″', 'ضریب 1 − η·L″'))} v={fmt(factor, 4)} testId="l12-factor" />
      </div>
      <P className="small muted">
        {Math.abs(factor) < 1
          ? factor < 0
            ? bi('Each step overshoots $w^*$ and lands on the other side, but closer: the run zig-zags in.', 'هر قدم از $w^*$ رد می‌شود و آن طرف می‌افتد، اما نزدیک‌تر: اجرا زیگزاگ‌وار نزدیک می‌شود.')
            : bi('Each step keeps the same side of $w^*$ and multiplies the distance by this factor.', 'هر قدم در همان سمت $w^*$ می‌ماند و فاصله را در این ضریب ضرب می‌کند.')
          : bi('|factor| ≥ 1: the distance to $w^*$ does not shrink — the run oscillates or diverges.', '|ضریب| ≥ ۱: فاصله تا $w^*$ کم نمی‌شود — اجرا نوسان می‌کند یا واگرا می‌شود.')}
      </P>
    </div>
  );
}

function Bottom() {
  const { tr } = useT();
  const c = useGD();
  const n = c.traj.length;
  const rows = c.traj.slice(-8).map((p, k) => ({ i: n - Math.min(8, n) + k, p }));
  const maxIter = Math.max(5, n - 1);
  return (
    <div className="viz-grid">
      <div>
        <div className="small muted">{tr(bi('Loss per step', 'خطا در هر گام'))}</div>
        <LossHistory height={200} />
      </div>
      <div>
        <div className="small muted">{tr(bi('Weight per step', 'وزن در هر گام'))}</div>
        <Plot2D
          testId="l12-w-history"
          xDomain={[0, maxIter]}
          yDomain="auto"
          height={200}
          xLabel={tr(bi('step', 'گام'))}
          yLabel="w"
          layers={[
            ...(Number.isFinite(c.ws) ? [{ kind: 'hline' as const, at: c.ws, color: 'var(--line-strong)', label: 'w*' }] : []),
            { kind: 'polyline', pts: c.traj.map((p, k) => [k, p[0]] as [number, number]), color: 'w', width: 2, markers: n < 60 },
          ]}
        />
      </div>
      <table className="tbl" data-testid="l12-table">
        <thead>
          <tr>
            <th>{tr(bi('step', 'گام'))}</th>
            <th className="num c-w">w</th>
            <th className="num c-L">L</th>
            <th className="num c-g">dL/dw</th>
            <th className="num c-lr">Δw</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ i, p }) => {
            const next = c.traj[i + 1];
            return (
              <tr key={i} className={i === n - 1 ? 'hl' : ''}>
                <td className="num">{i}</td>
                <td className="num">{fmt(p[0], 5)}</td>
                <td className="num">{fmt(p[1], 5)}</td>
                <td className="num">{fmt(c.gradAt(p[0]), 5)}</td>
                <td className="num">{next ? fmt(next[0] - p[0], 5) : '—'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export const lesson12: LessonDef = {
  id: ID,
  title: bi('Gradient Descent', 'گرادیان کاهشی (Gradient Descent)'),
  short: bi('Gradient descent', 'گرادیان کاهشی'),
  objectives: [
    bi('Explain why we step along the negative gradient', 'توضیح بدهی چرا در جهت منفی گرادیان قدم برمی‌داریم'),
    bi('Apply w_new = w_old − η·dL/dw with real numbers', 'قانون w_new = w_old − η·dL/dw را با عددهای واقعی به کار ببری'),
    bi('Watch repeated updates converge and the steps shrink near the minimum', 'ببینی به‌روزرسانی‌های پشت سر هم همگرا می‌شوند و قدم‌ها نزدیک کمینه کوچک می‌شوند'),
    bi('See how the learning rate changes the trajectory', 'ببینی نرخ یادگیری چطور مسیر را تغییر می‌دهد'),
  ],
  presetId: 'gd-step',
  revealSteps: 5,
  Explain,
  Center,
  Right,
  Bottom,
  checkpoint: {
    question: bi('At the current weight dL/dw = +12 and η = 0.1. What does one gradient-descent step do to w?', 'در وزن فعلی dL/dw = +12 و η = 0.1 است. یک گام گرادیان کاهشی با w چه می‌کند؟'),
    choices: [
      { id: 'a', label: bi('w increases by 1.2', 'w به اندازه ۱٫۲ زیاد می‌شود') },
      { id: 'b', label: bi('w decreases by 1.2', 'w به اندازه ۱٫۲ کم می‌شود') },
      { id: 'c', label: bi('w decreases by 12', 'w به اندازه ۱۲ کم می‌شود') },
    ],
    answer: 'b',
    explain: bi(
      'Δw = −η·dL/dw = −0.1 · 12 = −1.2. The slope is positive (uphill to the right), so we move left, by η times the slope.',
      'Δw = −η·dL/dw = −0.1 · 12 = −1.2. شیب مثبت است (به راست سربالایی)، پس به چپ می‌رویم، به اندازه η ضرب در شیب.',
    ),
  },
};
