// Lesson 17: Overfitting and generalization. Uses the shared lab store (preset "overfit": 8 noisy
// training points from sin(πx/2), 14 held-out validation points, 1→H tanh→1, Adam 0.03).
// The width H is rebuilt with Xavier init; training, both losses and the early-stopping point are
// read from the lab history. A complexity sweep trains one network per width with real steps.

import { useEffect, useMemo } from 'react';
import type { LessonDef } from './types';
import { bi, useT, type Bi } from '../i18n';
import { labTrainCfg, useLab } from '../state/lab';
import { useLessonState } from '../state/lessonState';
import { Plot2D, type PlotLayer } from '../components/Plot2D';
import { Tex } from '../components/Tex';
import { Slider, Stat } from '../components/controls';
import { Advanced, Callout, Cycle, P, Predict, Reveal } from '../components/teaching';
import { datasetLoss, fmt, newTrainState, overfitTruth, predict, unflatten } from '../engine';
import { bestValIndex, hiddenNet, sweepOne, verdict, type SweepRow, type Verdict } from './math/l17';

const ID = 'l17';
const SWEEP_STEPS = 1500;

interface S {
  seed: number;
  sweep: SweepRow[];
  sweeping: boolean;
}
const INIT: S = { seed: 4, sweep: [], sweeping: false };
const useS = () => useLessonState<S>(ID, INIT);

/** Fresh network of width H: new weights, history, schedule and optimizer state. */
function rebuild(H: number, seed: number) {
  const s = useLab.getState();
  s.setRunning(false);
  s.setNet(hiddenNet(H, seed));
  useLab.setState((st) => ({ train: newTrainState(st.net, st.data, labTrainCfg(st)) }));
}

const VERDICT: Record<Verdict, Bi> = {
  underfit: bi(
    'Underfitting: even the training points are fitted poorly. The model is too simple (or not trained enough).',
    'کم‌برازش (underfitting): حتی نقطه‌های آموزش هم بد برازش شده‌اند. مدل خیلی ساده است (یا به اندازه کافی آموزش ندیده).',
  ),
  good: bi(
    'Reasonable fit: training and validation losses are both low and close together.',
    'برازش معقول: خطای آموزش و اعتبارسنجی هر دو پایین و نزدیک به هم‌اند.',
  ),
  overfit: bi(
    'Overfitting: training loss is low but validation loss is much higher. The curve bends to chase the noise in the 8 training points.',
    'بیش‌برازش (overfitting): خطای آموزش پایین است ولی خطای اعتبارسنجی خیلی بیشتر است. منحنی برای دنبال کردن نویز ۸ نقطه آموزش پیچ‌وتاب می‌خورد.',
  ),
};

/** Current losses are computed from the current weights (so they also match restored weights). */
function useFit() {
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const val = useLab((s) => s.val);
  const loss = useLab((s) => s.loss);
  const lossOpts = useLab((s) => s.lossOpts);
  const iter = useLab((s) => s.train.schedule.iteration);
  const history = useLab((s) => s.history);
  const cur = useMemo(
    () => ({ iter, loss: datasetLoss(net, data, loss, lossOpts), val: val ? datasetLoss(net, val, loss, lossOpts) : undefined }),
    [net, data, val, loss, lossOpts, iter],
  );
  const best = bestValIndex(history);
  const H = net.layers[0].b.length;
  return { net, history, last: cur, best: best >= 0 ? history[best] : undefined, H, params: 3 * H + 1 };
}

function Explain() {
  const { tr } = useT();
  return (
    <>
      <Cycle active={2} />
      <Reveal at={1}>
        <h3>{tr(bi('1. The real goal: unseen data', '۱. هدف واقعی: داده‌ای که ندیده‌ایم'))}</h3>
        <P>
          {bi(
            'We train on 8 noisy measurements (filled dots) of a hidden true curve (dashed). But nobody needs predictions for points we already have. What matters is how well the network predicts new points, so 14 extra points (hollow) are kept aside as a validation set and never used for training.',
            'روی ۸ اندازه‌گیری نویزدار (نقطه‌های توپر) از یک منحنی واقعی پنهان (خط‌چین) آموزش می‌دهیم. ولی کسی به پیش‌بینی نقطه‌هایی که از قبل داریم نیاز ندارد. مهم این است که شبکه نقطه‌های جدید را چقدر خوب پیش‌بینی می‌کند؛ برای همین ۱۴ نقطه دیگر (توخالی) به عنوان مجموعه اعتبارسنجی (validation set) کنار گذاشته شده‌اند و هیچ‌وقت در آموزش استفاده نمی‌شوند.',
          )}
        </P>
        <P>
          {bi(
            'Set hidden units to 1 and train: one tanh can only make one S-shaped bend, so both losses stay high. That is underfitting.',
            'تعداد نورون‌های پنهان را ۱ بگذار و آموزش بده: یک tanh فقط یک خم S شکل می‌سازد، پس هر دو خطا بالا می‌مانند. این کم‌برازش (underfitting) است.',
          )}
        </P>
      </Reveal>
      <Reveal at={2}>
        <Predict
          id={ID + '-val'}
          question={bi('Train a 10-unit network for a long time. Its training loss drops to almost 0. What does its validation loss do?', 'یک شبکه ۱۰ نورونی را مدت طولانی آموزش بده. خطای آموزشش تقریباً به ۰ می‌رسد. خطای اعتبارسنجی‌اش چه می‌کند؟')}
          choices={[
            { id: 'a', label: bi('Also drops to almost 0', 'آن هم تقریباً ۰ می‌شود') },
            { id: 'b', label: bi('Reaches a minimum, then rises', 'به یک کمینه می‌رسد و بعد بالا می‌رود') },
            { id: 'c', label: bi('Stays exactly equal to the training loss', 'دقیقاً برابر خطای آموزش می‌ماند') },
          ]}
          answer="b"
          explain={bi(
            'Watch the two curves at the bottom: the validation loss (dashed) bottoms out early and then climbs while the training loss keeps falling. The network starts memorizing the noise of the 8 points.',
            'دو منحنی پایین را نگاه کن: خطای اعتبارسنجی (خط‌چین) زود به کف می‌رسد و بعد بالا می‌رود، در حالی که خطای آموزش همچنان پایین می‌آید. شبکه شروع می‌کند به حفظ کردن نویز ۸ نقطه.',
          )}
        />
      </Reveal>
      <Reveal at={3}>
        <h3>{tr(bi('2. Lower training loss is not better', '۲. خطای آموزش کمتر یعنی بهتر نیست'))}</h3>
        <P>
          {bi(
            'With 10 hidden units the network has 31 parameters for 8 points: enough freedom to pass through every training point exactly, noise included. Between the points the curve is free to swing wildly, and the validation points expose it. Training loss measures memory; validation loss measures generalization.',
            'با ۱۰ نورون پنهان، شبکه ۳۱ پارامتر برای ۸ نقطه دارد: آن‌قدر آزادی که دقیقاً از همه نقطه‌های آموزش رد شود، نویزشان هم حساب. بین نقطه‌ها منحنی آزاد است هر طور بخواهد تاب بخورد و نقطه‌های اعتبارسنجی این را لو می‌دهند. خطای آموزش «حافظه» را می‌سنجد؛ خطای اعتبارسنجی «تعمیم» (generalization) را.',
          )}
        </P>
        <Callout>
          {tr(
            bi(
              'Run the complexity sweep at the bottom: training loss keeps falling as the network grows, but validation loss is lowest for a middle size.',
              'جاروب پیچیدگی (complexity sweep) پایین را اجرا کن: با بزرگ شدن شبکه خطای آموزش مدام کم می‌شود، ولی کمترین خطای اعتبارسنجی مال یک اندازه متوسط است.',
            ),
          )}
        </Callout>
      </Reveal>
      <Reveal at={4}>
        <h3>{tr(bi('3. Early stopping', '۳. توقف زودهنگام (early stopping)'))}</h3>
        <P>
          {bi(
            'A simple fix: keep checking the validation loss while training and remember the weights from the step where it was lowest (the marked line in the loss plot). "Restore best weights" on the right does exactly that. Other remedies: more data, a smaller model, or regularization such as weight decay and dropout.',
            'یک راه ساده: در طول آموزش مدام خطای اعتبارسنجی را بررسی کن و وزن‌های قدمی را که کمترین مقدار را داشت نگه دار (خط علامت‌خورده در نمودار خطا). دکمه «برگرداندن بهترین وزن‌ها» در سمت راست دقیقاً همین کار را می‌کند. راه‌های دیگر: داده بیشتر، مدل کوچک‌تر، یا منظم‌سازی (regularization) مثل weight decay و dropout.',
          )}
        </P>
      </Reveal>
      <Reveal at={5}>
        <Advanced title={bi('Bias, variance and the noise floor', 'سوگیری، واریانس و کف نویز')}>
          <P>
            {bi(
              'For a new point with $y = f(x) + \\epsilon$, $\\operatorname{Var}(\\epsilon) = \\sigma^2$, the expected squared error over training sets splits into $\\text{bias}^2 + \\text{variance} + \\sigma^2$. Small models have high bias (they cannot bend enough); large models trained to zero training error have high variance (a different noisy sample would give a very different curve). The validation noise here has $\\sigma = 0.25$, so even the true function scores about $\\sigma^2 \\approx 0.06$: no model can reliably beat that.',
              'برای یک نقطه جدید با $y = f(x) + \\epsilon$ و $\\operatorname{Var}(\\epsilon) = \\sigma^2$، امید خطای مربعی روی مجموعه‌های آموزش مختلف به $\\text{bias}^2 + \\text{variance} + \\sigma^2$ تجزیه می‌شود. مدل‌های کوچک سوگیری (bias) بالایی دارند (نمی‌توانند به اندازه کافی خم شوند)؛ مدل‌های بزرگی که تا خطای آموزش صفر آموزش دیده‌اند واریانس بالایی دارند (یک نمونه نویزدار دیگر منحنی خیلی متفاوتی می‌داد). نویز اعتبارسنجی اینجا $\\sigma = 0.25$ است، پس حتی خود تابع واقعی حدود $\\sigma^2 \\approx 0.06$ خطا می‌گیرد: هیچ مدلی نمی‌تواند به‌طور قابل اعتماد بهتر از این باشد.',
            )}
          </P>
          <P>
            {bi(
              'Choosing the width or the stopping step by validation loss makes that loss slightly optimistic, because the choice was fitted to it. A third, untouched test set gives the honest final number.',
              'انتخاب پهنا یا قدم توقف بر اساس خطای اعتبارسنجی آن خطا را کمی خوش‌بینانه می‌کند، چون انتخاب روی همان داده تنظیم شده است. یک مجموعه سوم دست‌نخورده (test set) عدد نهایی صادقانه را می‌دهد.',
            )}
          </P>
        </Advanced>
      </Reveal>
    </>
  );
}

function Center() {
  const { tr } = useT();
  const [s] = useS();
  const data = useLab((st) => st.data);
  const val = useLab((st) => st.val);
  const running = useLab((st) => st.running);
  const setRunning = useLab((st) => st.setRunning);
  const trainSteps = useLab((st) => st.trainSteps);
  const { net, last, H } = useFit();
  const layers = useMemo<PlotLayer[]>(
    () => [
      { kind: 'fn', f: overfitTruth, color: 'var(--ink-2)', width: 1.6, dash: '6 5' },
      { kind: 'fn', f: (x) => predict(net, [x])[0], color: 'yhat', width: 2.8 },
      { kind: 'points', color: 'x', pts: (val?.X ?? []).map((x, n) => ({ x: x[0], y: val!.Y[n][0], r: 4.5, hollow: true })) },
      { kind: 'points', color: 'y', pts: data.X.map((x, n) => ({ x: x[0], y: data.Y[n][0], r: 5.5 })) },
    ],
    [net, data, val],
  );
  const v = last?.val !== undefined ? verdict(last.loss, last.val) : 'good';
  return (
    <div className="viz">
      <div className="row">
        <div className="grow">
          <Slider label={tr(bi('hidden units H', 'نورون پنهان H'))} value={H} min={1} max={10} step={1} digits={0} color="a" onChange={(h) => rebuild(Math.round(h), s.seed)} testId="of-units" />
        </div>
        <button className={`btn ${running ? '' : 'primary'}`} onClick={() => setRunning(!running)} data-testid="of-train">
          {running ? `❚❚ ${tr(bi('Pause', 'توقف'))}` : `▶ ${tr(bi('Train', 'آموزش'))}`}
        </button>
        <button className="btn" onClick={() => trainSteps(200)} disabled={running} data-testid="of-200">
          +200
        </button>
        <button className="btn ghost" onClick={() => rebuild(H, s.seed)} data-testid="of-reset">
          ↺
        </button>
      </div>
      <div className="legend-chips">
        <span className="chip"><i style={{ background: 'var(--c-y)' }} /> {tr(bi('training points (8)', 'نقطه‌های آموزش (۸)'))}</span>
        <span className="chip"><i style={{ background: 'transparent', border: '2px solid var(--c-x)' }} /> {tr(bi('validation points (14, never trained on)', 'نقطه‌های اعتبارسنجی (۱۴، هرگز در آموزش نبوده‌اند)'))}</span>
        <span className="chip"><i style={{ background: 'var(--c-yhat)' }} /> {tr(bi('network ŷ(x)', 'شبکه ŷ(x)'))}</span>
        <span className="chip"><i style={{ background: 'var(--ink-2)' }} /> {tr(bi('true function sin(πx/2)', 'تابع واقعی sin(πx/2)'))}</span>
      </div>
      <Plot2D testId="of-plot" xDomain={[-2.3, 2.3]} yDomain={[-2, 2]} layers={layers} height={340} xLabel="x" yLabel="y" />
      {last?.val !== undefined && (
        <Callout tone={v === 'good' ? 'info' : 'warn'}>
          <strong data-testid="of-verdict">{tr(bi(`H = ${H}, iteration ${last.iter}: `, `H = ${H}، تکرار ${last.iter}: `))}</strong>
          {tr(VERDICT[v])}
        </Callout>
      )}
    </div>
  );
}

function Right() {
  const { tr } = useT();
  const [s, set] = useS();
  const opt = useLab((st) => st.opt);
  const setOpt = useLab((st) => st.setOpt);
  const setNet = useLab((st) => st.setNet);
  const setRunning = useLab((st) => st.setRunning);
  const { net, last, best, H, params } = useFit();
  return (
    <div className="stack">
      <Slider label={tr(bi('init seed', 'بذر مقداردهی'))} value={s.seed} min={1} max={30} step={1} digits={0} onChange={(seed) => (set({ seed }), rebuild(H, seed))} />
      <Slider label={<Tex>{'\\clr{\\eta}\\ (\\text{Adam})'}</Tex>} value={opt.lr} min={0.001} max={0.1} step={0.001} color="lr" digits={4} onChange={(lr) => setOpt({ lr })} />
      <Tex display>{'\\cL{L_{\\text{train}}} = \\tfrac{1}{8}\\sum_{n=1}^{8}(\\cyhat{\\hat y_n}-\\cy{y_n})^2 \\qquad \\cL{L_{\\text{val}}} = \\tfrac{1}{14}\\sum_{m=1}^{14}(\\cyhat{\\hat y_m}-\\cy{y_m})^2'}</Tex>
      <div className="stat-grid">
        <Stat k={tr(bi('parameters', 'پارامترها'))} v={`3·${H}+1 = ${params}`} />
        <Stat k={tr(bi('training points', 'نقطه آموزش'))} v={8} />
        <Stat k={tr(bi('iteration', 'تکرار'))} v={last?.iter ?? 0} />
        <Stat k={tr(bi('train loss', 'خطای آموزش'))} v={fmt(last?.loss ?? NaN, 4)} color="L" testId="of-train-loss" />
        <Stat k={tr(bi('validation loss', 'خطای اعتبارسنجی'))} v={fmt(last?.val ?? NaN, 4)} color="y" testId="of-val-loss" />
        <Stat k={tr(bi('gap (val − train)', 'فاصله (val − train)'))} v={fmt((last?.val ?? NaN) - (last?.loss ?? NaN), 4)} />
      </div>
      {params > 8 && (
        <P className="small muted">
          {bi(`${params} parameters for 8 points: enough freedom to memorize them.`, `${params} پارامتر برای ۸ نقطه: آن‌قدر آزادی که بتواند همه را حفظ کند.`)}
        </P>
      )}
      <h4>{tr(bi('Early stopping', 'توقف زودهنگام'))}</h4>
      {best && best.val !== undefined ? (
        <>
          <div className="stat-grid">
            <Stat k={tr(bi('best val loss', 'بهترین خطای val'))} v={fmt(best.val, 4)} color="y" testId="of-best" />
            <Stat k={tr(bi('at iteration', 'در تکرار'))} v={best.iter} />
            <Stat k={tr(bi('train loss then', 'خطای آموزش آن موقع'))} v={fmt(best.loss, 4)} color="L" />
          </div>
          <button
            className="btn"
            disabled={!best.theta}
            onClick={() => {
              setRunning(false);
              if (best.theta) setNet(unflatten(net, best.theta), true);
            }}
            data-testid="of-restore"
          >
            ⤺ {tr(bi(`Restore best weights (iteration ${best.iter})`, `برگرداندن بهترین وزن‌ها (تکرار ${best.iter})`))}
          </button>
        </>
      ) : (
        <P className="small muted">{bi('Train to see where the validation loss is lowest.', 'آموزش بده تا ببینی خطای اعتبارسنجی کجا کمترین است.')}</P>
      )}
    </div>
  );
}

function Sweep() {
  const { tr } = useT();
  const [s, set] = useS();
  const data = useLab((st) => st.data);
  const val = useLab((st) => st.val);
  // One width per animation frame keeps the UI responsive (10 × 1500 Adam steps in total).
  useEffect(() => {
    if (!s.sweeping || !val) return;
    const raf = requestAnimationFrame(() => {
      const st = useLab.getState();
      const done = s.sweep.length;
      const row = sweepOne(done + 1, s.seed, SWEEP_STEPS, data, val, { loss: st.loss, lossOpts: st.lossOpts, opt: st.opt, batch: st.batch, divergeAt: 1e8 });
      set({ sweep: [...s.sweep, row], sweeping: done + 1 < 10 });
    });
    return () => cancelAnimationFrame(raf);
  }, [s.sweeping, s.sweep, s.seed, data, val, set]);
  const pts = (f: (r: SweepRow) => number) => s.sweep.map((r) => [r.H, f(r)] as [number, number]);
  const layers: PlotLayer[] = [
    { kind: 'polyline', pts: pts((r) => r.train), color: 'L', width: 2.2, markers: true },
    { kind: 'polyline', pts: pts((r) => r.val), color: 'y', width: 2.2, dash: '6 4', markers: true },
    { kind: 'polyline', pts: pts((r) => r.bestVal), color: 'x', width: 1.6, dash: '2 3', markers: true },
  ];
  return (
    <div>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <span className="small muted">{tr(bi(`Complexity sweep: H = 1…10, ${SWEEP_STEPS} Adam steps each`, `جاروب پیچیدگی: H = 1…10، هر کدام ${SWEEP_STEPS} قدم Adam`))}</span>
        <button className="btn sm" disabled={s.sweeping} onClick={() => set({ sweep: [], sweeping: true })} data-testid="of-sweep">
          {s.sweeping ? `${s.sweep.length}/10 …` : tr(bi('Run sweep', 'اجرای جاروب'))}
        </button>
      </div>
      <Plot2D
        testId="of-sweep-plot"
        xDomain={[0.5, 10.5]}
        yDomain={[0, Math.max(0.3, ...s.sweep.map((r) => Math.min(1, r.val)))]}
        layers={layers}
        height={200}
        xLabel={tr(bi('hidden units H', 'نورون پنهان H'))}
        yLabel="L"
        legend={[
          { label: tr(bi('train (final)', 'آموزش (پایانی)')), color: 'L' },
          { label: tr(bi('validation (final)', 'اعتبارسنجی (پایانی)')), color: 'y', dash: true },
          { label: tr(bi('validation (best, early stop)', 'اعتبارسنجی (بهترین، توقف زودهنگام)')), color: 'x', dash: true },
        ]}
      />
    </div>
  );
}

function Bottom() {
  const { tr } = useT();
  const { history, best } = useFit();
  const layers = useMemo<PlotLayer[]>(() => {
    const step = Math.max(1, Math.floor(history.length / 600));
    const pts = history.filter((_, i) => i % step === 0 || i === history.length - 1);
    const out: PlotLayer[] = [
      { kind: 'polyline', pts: pts.map((h) => [h.iter, h.loss]), color: 'L', width: 2 },
      { kind: 'polyline', pts: pts.filter((h) => h.val !== undefined).map((h) => [h.iter, h.val!]), color: 'y', width: 2, dash: '6 4' },
    ];
    if (best) out.push({ kind: 'vline', at: best.iter, color: 'x', label: tr(bi('lowest val → stop here', 'کمترین val ← اینجا توقف')) });
    return out;
  }, [history, best, tr]);
  const maxIter = Math.max(10, history[history.length - 1]?.iter ?? 10);
  const top = Math.min(1.2, Math.max(0.2, ...history.slice(0, 50).map((h) => Math.max(h.loss, h.val ?? 0))));
  return (
    <div className="viz-grid">
      <div>
        <div className="small muted">{tr(bi('Training loss (solid) and validation loss (dashed) during training', 'خطای آموزش (خط پر) و خطای اعتبارسنجی (خط‌چین) در طول آموزش'))}</div>
        <Plot2D testId="of-loss" xDomain={[history[0]?.iter ?? 0, maxIter]} yDomain={[0, top * 1.05]} layers={layers} height={200} xLabel={tr(bi('iteration', 'تکرار'))} yLabel="L" />
      </div>
      <Sweep />
    </div>
  );
}

export const lesson17: LessonDef = {
  id: ID,
  title: bi('Overfitting and Generalization', 'بیش‌برازش و تعمیم'),
  short: bi('Overfitting', 'بیش‌برازش'),
  objectives: [
    bi('Separate training data from held-out validation data', 'داده آموزش را از داده اعتبارسنجی کنارگذاشته جدا کنی'),
    bi('Recognize underfitting, a reasonable fit and overfitting', 'کم‌برازش، برازش معقول و بیش‌برازش را تشخیص بدهی'),
    bi('Explain why lower training loss does not mean better generalization', 'توضیح بدهی چرا خطای آموزش کمتر به معنی تعمیم بهتر نیست'),
    bi('Use early stopping at the lowest validation loss', 'از توقف زودهنگام در کمترین خطای اعتبارسنجی استفاده کنی'),
  ],
  presetId: 'overfit',
  revealSteps: 5,
  Explain,
  Center,
  Right,
  Bottom,
  checkpoint: {
    question: bi('Model A: training loss 0.001, validation loss 0.16. Model B: training loss 0.05, validation loss 0.07. Which one should you deploy?', 'مدل A: خطای آموزش 0.001، خطای اعتبارسنجی 0.16. مدل B: خطای آموزش 0.05، خطای اعتبارسنجی 0.07. کدام را به کار می‌بری؟'),
    choices: [
      { id: 'a', label: bi('A: it fits the training data almost perfectly', 'A: تقریباً بی‌نقص روی داده آموزش برازش شده') },
      { id: 'b', label: bi('B: it does better on data it has not seen', 'B: روی داده‌ای که ندیده بهتر عمل می‌کند') },
      { id: 'c', label: bi('They are equally good', 'هر دو به یک اندازه خوب‌اند') },
    ],
    answer: 'b',
    explain: bi(
      'New inputs behave like validation data, not training data. A\'s tiny training loss is memorized noise; B generalizes better. Compare H = 10 trained long with H = 3 in the demo.',
      'ورودی‌های جدید شبیه داده اعتبارسنجی‌اند، نه داده آموزش. خطای آموزش ناچیز A یعنی نویز را حفظ کرده؛ B بهتر تعمیم می‌دهد. در دمو H = 10 با آموزش طولانی را با H = 3 مقایسه کن.',
    ),
  },
};
