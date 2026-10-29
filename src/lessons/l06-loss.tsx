// Lesson 06: Predictions and loss. Starts from preset "temperature" (outside temperature -> cooling
// power, 4 examples, one linear neuron in the shared lab store): errors, squared errors, MSE, then
// MAE and Huber with an outlier, BCE for "will it rain?" and CCE for 3-class weather. Every loss
// value and gradient comes from the engine (batchLossGrad / lossSample / lossGrad / softmax).

import { useMemo } from 'react';
import type { LessonDef } from './types';
import { bi, useT, type Bi } from '../i18n';
import { useLab } from '../state/lab';
import { useLessonState } from '../state/lessonState';
import { Plot2D, type PlotLayer } from '../components/Plot2D';
import { Tex } from '../components/Tex';
import { Segmented, Slider, Stat } from '../components/controls';
import { Advanced, Callout, Cycle, P, Predict, Reveal } from '../components/teaching';
import { backLayer, batchLossGrad, clamp, fmt, fmtSigned, lossGrad, lossSample, predict, softmax, tex, texP } from '../engine';
import { fitLine, OUTLIER, withOutlier, type RegLoss } from './math/l06';

const ID = 'l06';

type Stage = 'reg' | 'bce' | 'cce';
interface Day {
  p: number;
  y: 0 | 1;
}
interface S {
  stage: Stage;
  loss: RegLoss;
  delta: number;
  outlier: boolean;
  best: boolean;
  focus: number;
  days: Day[];
  day: number;
  logits: number[];
  cls: number;
}
const INIT: S = {
  stage: 'reg',
  loss: 'mse',
  delta: 0.5,
  outlier: false,
  best: false,
  focus: 0,
  days: [
    { p: 0.8, y: 1 },
    { p: 0.3, y: 0 },
    { p: 0.6, y: 0 },
    { p: 0.1, y: 1 },
  ],
  day: 0,
  logits: [2, 1, 0.2],
  cls: 2,
};
const useS = () => useLessonState<S>(ID, INIT);

const CLASSES = [bi('sunny', 'آفتابی'), bi('cloudy', 'ابری'), bi('rainy', 'بارانی')];
const LOSS_NAME: Record<RegLoss, string> = { mse: 'MSE', mae: 'MAE', huber: 'Huber' };
const TERM: Record<RegLoss, (n: number) => string> = { mse: (n) => `e_${n}^2`, mae: (n) => `|e_${n}|`, huber: (n) => `h_\\delta(e_${n})` };
const W_REF = { kind: 'W' as const, layer: 0, j: 0, i: 0 };
const B_REF = { kind: 'b' as const, layer: 0, j: 0, i: -1 };

/** The regression stage: lab network and data (plus the optional outlier), losses from the engine. */
function useReg() {
  const [s] = useS();
  const net = useLab((st) => st.net);
  const base = useLab((st) => st.data);
  return useMemo(() => {
    const data = withOutlier(base, s.outlier);
    const o = { delta: s.delta };
    const res = batchLossGrad(net, data, s.loss, o);
    const rows = data.X.map((x, n) => {
      const yhat = predict(net, x)[0];
      const y = data.Y[n][0];
      return { x: x[0], y, yhat, e: yhat - y, l: res.perSample[n], g: lossGrad(s.loss, [yhat], [y], o)[0], outlier: s.outlier && n === base.X.length };
    });
    return {
      net,
      data,
      rows,
      N: rows.length,
      L: res.loss,
      gw: res.grads.W[0][0][0],
      gb: res.grads.b[0][0],
      w: net.layers[0].W[0][0],
      b: net.layers[0].b[0],
      best: fitLine(s.loss, data, s.delta),
    };
  }, [net, base, s.outlier, s.delta, s.loss]);
}

const bceL = (d: Day) => lossSample('bce', [d.p], [d.y]);
const bceG = (d: Day) => lossGrad('bce', [d.p], [d.y])[0];

function useCce() {
  const [s] = useS();
  return useMemo(() => {
    const p = softmax(s.logits);
    const y = s.logits.map((_, k) => (k === s.cls ? 1 : 0));
    const dp = lossGrad('cce', p, y);
    return { p, y, loss: lossSample('cce', p, y), dp, dz: backLayer('softmax', s.logits, p, dp) };
  }, [s.logits, s.cls]);
}

function Explain() {
  const { tr } = useT();
  const [s, set] = useS();
  return (
    <>
      <Cycle active={s.stage === 'reg' ? 1 : 3} />
      <Reveal at={1}>
        <h3>{tr(bi('1. Prediction versus target', '۱. پیش‌بینی در برابر هدف'))}</h3>
        <P>
          {bi(
            'An air conditioner company wants to predict the cooling power a building needs from the outside temperature. One neuron predicts $\\cyhat{\\hat y} = \\cw{w}\\cx{x} + \\cb{b}$. For four past days we know the real answer, the target $\\cy{y}$.',
            'یک شرکت کولرسازی می‌خواهد از روی دمای بیرون، توان سرمایشی لازم برای یک ساختمان را پیش‌بینی کند. یک نورون پیش‌بینی می‌کند: $\\cyhat{\\hat y} = \\cw{w}\\cx{x} + \\cb{b}$. برای چهار روز گذشته جواب واقعی را می‌دانیم، یعنی هدف (target) $\\cy{y}$.',
          )}
        </P>
        <P>
          {bi(
            'The error of one example is $e = \\cyhat{\\hat y} - \\cy{y}$: positive means the prediction was too high, negative too low. The red segments on the plot are these errors.',
            'خطای (error) یک مثال $e = \\cyhat{\\hat y} - \\cy{y}$ است: مثبت یعنی پیش‌بینی زیادی بالا بوده، منفی یعنی زیادی پایین. پاره‌خط‌های قرمز روی نمودار همین خطاها هستند.',
          )}
        </P>
      </Reveal>
      <Reveal at={2}>
        <h3>{tr(bi('2. Squared error and MSE', '۲. مربع خطا و MSE'))}</h3>
        <P>
          {bi(
            'To score the model with one number, square each error and average: the Mean Squared Error. Squaring stops positive and negative errors from cancelling, and it makes a big miss count much more than a small one.',
            'برای اینکه به مدل یک نمره واحد بدهیم، هر خطا را به توان ۲ می‌رسانیم و میانگین می‌گیریم: میانگین مربع خطا (Mean Squared Error). توان ۲ نمی‌گذارد خطاهای مثبت و منفی همدیگر را خنثی کنند، و باعث می‌شود یک اشتباه بزرگ خیلی بیشتر از یک اشتباه کوچک حساب شود.',
          )}
        </P>
        <Tex display>{'\\cL{\\text{MSE}} = \\frac{1}{N}\\sum_{n=1}^{N} (\\cyhat{\\hat y_n} - \\cy{y_n})^2'}</Tex>
        <Predict
          id={ID + '-w'}
          question={bi('At the start ($w = 1$, $b = -1.2$), if you increase $w$ a little, does the MSE go up or down?', 'در شروع ($w = 1$ و $b = -1.2$)، اگر $w$ را کمی زیاد کنی، MSE بالا می‌رود یا پایین؟')}
          choices={[
            { id: 'up', label: bi('Up', 'بالا') },
            { id: 'down', label: bi('Down', 'پایین') },
          ]}
          answer="down"
          explain={bi(
            'The line is too low on the hot days (negative errors at large $x$). A larger $w$ lifts the right side more than the left, so those errors shrink. The gradient $\\partial L/\\partial w$ on the right is negative, which says exactly this.',
            'خط در روزهای گرم زیادی پایین است (خطای منفی در $x$های بزرگ). $w$ بزرگ‌تر سمت راست خط را بیشتر از سمت چپ بالا می‌برد، پس آن خطاها کوچک می‌شوند. گرادیان $\\partial L/\\partial w$ در سمت راست منفی است و دقیقاً همین را می‌گوید.',
          )}
        />
      </Reveal>
      <Reveal at={3}>
        <h3>{tr(bi('3. Loss value versus loss gradient', '۳. مقدار خطا در برابر گرادیان خطا'))}</h3>
        <P>
          {bi(
            'The loss value answers "how bad is it?". The gradient answers "which way, and how strongly, does the loss change if I nudge something?". For one example, $\\partial (e^2)/\\partial \\hat y = 2e$: a prediction that is too high gets a positive gradient (move it down). For the parameters, $\\partial L/\\partial w$ and $\\partial L/\\partial b$ tell you which way to turn each knob.',
            'مقدار خطا (loss value) جواب «چقدر بد است؟» را می‌دهد. گرادیان (gradient) جواب «اگر چیزی را کمی تکان بدهم، خطا به کدام سمت و با چه شدتی عوض می‌شود؟» را. برای یک مثال، $\\partial (e^2)/\\partial \\hat y = 2e$: پیش‌بینی‌ای که زیادی بالاست گرادیان مثبت می‌گیرد (باید پایین بیاید). برای پارامترها، $\\partial L/\\partial w$ و $\\partial L/\\partial b$ می‌گویند هر پیچ را به کدام سمت بچرخانی.',
          )}
        </P>
        <P>
          {bi(
            'They are different things: at the best line the value is still above 0 but the gradient is 0, and MAE far from the target has a large value but a gradient of only $\\pm 1/N$.',
            'این دو چیز متفاوتی‌اند: روی بهترین خط مقدار خطا هنوز بیشتر از ۰ است ولی گرادیان ۰ است، و MAE دور از هدف مقدار بزرگی دارد ولی گرادیانش فقط $\\pm 1/N$ است.',
          )}
        </P>
      </Reveal>
      <Reveal at={4}>
        <h3>{tr(bi('4. MAE and Huber: robust to outliers', '۴. MAE و Huber: مقاوم در برابر داده پرت'))}</h3>
        <P>
          {bi(
            'Mean Absolute Error averages $|e|$ instead of $e^2$. Huber is $\\tfrac12 e^2$ for small errors ($|e| \\le \\delta$) and grows only linearly beyond $\\delta$. Turn on the broken-sensor outlier and "show best line": the MSE line bends toward the outlier, MAE and Huber barely move.',
            'میانگین قدر مطلق خطا (Mean Absolute Error) به جای $e^2$ میانگین $|e|$ را می‌گیرد. Huber برای خطاهای کوچک ($|e| \\le \\delta$) برابر $\\tfrac12 e^2$ است و بعد از $\\delta$ فقط خطی رشد می‌کند. داده پرت (سنسور خراب) و «نمایش بهترین خط» را روشن کن: خط MSE به سمت داده پرت کج می‌شود، ولی MAE و Huber تقریباً تکان نمی‌خورند.',
          )}
        </P>
      </Reveal>
      <Reveal at={5}>
        <h3>{tr(bi('5. Will it rain? Binary cross-entropy', '۵. باران می‌آید؟ آنتروپی متقاطع دودویی'))}</h3>
        <P>
          {bi(
            'Now the model outputs a probability $p$ that it rains, and the label is $y = 1$ (rain) or $y = 0$ (no rain). Binary Cross-Entropy is $-\\log p$ on rainy days and $-\\log(1-p)$ on dry days: small when the model is confident and right, huge when it is confident and wrong.',
            'حالا مدل احتمال $p$ بارش باران را می‌دهد و برچسب $y = 1$ (باران) یا $y = 0$ (بدون باران) است. آنتروپی متقاطع دودویی (Binary Cross-Entropy) در روزهای بارانی $-\\log p$ و در روزهای خشک $-\\log(1-p)$ است: وقتی مدل مطمئن و درست است کوچک است، وقتی مطمئن و غلط است خیلی بزرگ.',
          )}
        </P>
        {s.stage !== 'bce' && (
          <button className="btn sm" onClick={() => set({ stage: 'bce' })} data-testid="to-bce">
            {tr(bi('Try the rain forecast →', 'پیش‌بینی باران را امتحان کن ←'))}
          </button>
        )}
      </Reveal>
      <Reveal at={6}>
        <h3>{tr(bi('6. Three kinds of weather: categorical cross-entropy', '۶. سه نوع هوا: آنتروپی متقاطع چندکلاسه'))}</h3>
        <P>
          {bi(
            'With several classes (sunny, cloudy, rainy) the network outputs one score per class and softmax turns the scores into probabilities that add up to 1. Categorical Cross-Entropy only looks at the probability of the true class: $\\ell = -\\log p_{\\text{true}}$.',
            'وقتی چند کلاس داریم (آفتابی، ابری، بارانی)، شبکه برای هر کلاس یک امتیاز می‌دهد و softmax امتیازها را به احتمال‌هایی تبدیل می‌کند که جمعشان ۱ است. آنتروپی متقاطع چندکلاسه (Categorical Cross-Entropy) فقط به احتمال کلاس درست نگاه می‌کند: $\\ell = -\\log p_{\\text{true}}$.',
          )}
        </P>
        {s.stage !== 'cce' && (
          <button className="btn sm" onClick={() => set({ stage: 'cce' })} data-testid="to-cce">
            {tr(bi('Try the 3-class weather →', 'هوای سه‌کلاسه را امتحان کن ←'))}
          </button>
        )}
      </Reveal>
      <Reveal at={7}>
        <Advanced>
          <P>
            {bi(
              'Normalization used by the engine: the per-sample loss averages over output units, except CCE, which sums over classes; the batch loss is the mean of per-sample losses. So the gradient of the batch MSE with respect to one prediction is $2e_n/N$.',
              'قرارداد نرمال‌سازی موتور: خطای هر نمونه روی واحدهای خروجی میانگین گرفته می‌شود، به جز CCE که روی کلاس‌ها جمع می‌زند؛ خطای دسته (batch) میانگین خطای نمونه‌هاست. پس گرادیان MSE دسته نسبت به یک پیش‌بینی برابر $2e_n/N$ است.',
            )}
          </P>
          <P>
            {bi(
              'Gradients with respect to the prediction: MSE $2e$, MAE $\\operatorname{sign}(e)$ (the engine uses $\\operatorname{sign}(0) = 0$ at the kink), Huber $\\operatorname{clip}(e, -\\delta, \\delta)$, BCE $\\frac{p-y}{p(1-p)}$, CCE $-y_k/p_k$. With a sigmoid or softmax output these simplify to $p - y$ with respect to the score $z$. Probabilities are clipped to $[10^{-12}, 1 - 10^{-12}]$ so $\\log 0$ never occurs.',
              'گرادیان‌ها نسبت به پیش‌بینی: MSE برابر $2e$، MAE برابر $\\operatorname{sign}(e)$ (موتور در نقطه شکست $\\operatorname{sign}(0) = 0$ را استفاده می‌کند)، Huber برابر $\\operatorname{clip}(e, -\\delta, \\delta)$، BCE برابر $\\frac{p-y}{p(1-p)}$ و CCE برابر $-y_k/p_k$. با خروجی sigmoid یا softmax، گرادیان نسبت به امتیاز $z$ ساده می‌شود به $p - y$. احتمال‌ها به بازه $[10^{-12}, 1 - 10^{-12}]$ بریده می‌شوند تا هیچ‌وقت $\\log 0$ پیش نیاید.',
            )}
          </P>
          <P>
            {bi(
              'For a constant prediction $c$, MSE is minimized by the mean of the targets and MAE by their median. That is why one extreme value drags MSE much further.',
              'برای یک پیش‌بینی ثابت $c$، کمینه MSE در میانگین هدف‌ها و کمینه MAE در میانه (median) آنهاست. برای همین یک مقدار خیلی پرت MSE را خیلی بیشتر با خودش می‌کشد.',
            )}
          </P>
        </Advanced>
      </Reveal>
    </>
  );
}

function RegCenter() {
  const { tr } = useT();
  const [s, set] = useS();
  const r = useReg();
  const setParam = useLab((st) => st.setParam);
  const layers = useMemo<PlotLayer[]>(() => {
    const out: PlotLayer[] = [];
    if (s.best) out.push({ kind: 'line', p: [0, r.best.b], slope: r.best.w, color: 'g', dash: '7 5', width: 1.8 });
    out.push({ kind: 'fn', f: (x) => predict(r.net, [x])[0], color: 'yhat', width: 2.8 });
    r.rows.forEach((q, n) =>
      out.push({ kind: 'segment', from: [q.x, q.y], to: [q.x, q.yhat], color: 'L', width: n === s.focus ? 3.4 : 2.2, label: `e=${fmtSigned(q.e, 2)}` }),
    );
    out.push(
      { kind: 'points', color: 'yhat', pts: r.rows.map((q) => ({ x: q.x, y: q.yhat, r: 3.5, hollow: true })) },
      { kind: 'points', color: 'y', pts: r.rows.map((q) => ({ x: q.x, y: q.y, r: q.outlier ? 7 : 5, label: q.outlier ? tr(bi('outlier', 'داده پرت')) : undefined })) },
    );
    return out;
  }, [r, s.best, s.focus, tr]);
  return (
    <>
      <Plot2D
        testId="reg-plot"
        xDomain={[1.6, 3.9]}
        yDomain={[-0.3, 3.6]}
        layers={layers}
        height={340}
        xLabel={tr(bi('outside temperature (×10 °C)', 'دمای بیرون (×۱۰ درجه)'))}
        yLabel={tr(bi('cooling power (kW)', 'توان سرمایش (کیلووات)'))}
        onDrag={(x, y) => setParam(B_REF, Math.round((y - r.w * x) * 100) / 100)}
        legend={[
          { label: tr(bi('prediction ŷ = w·x + b', 'پیش‌بینی ŷ = w·x + b')), color: 'yhat' },
          { label: tr(bi('target y', 'هدف y')), color: 'y' },
          { label: tr(bi('error e = ŷ − y', 'خطا e = ŷ − y')), color: 'L' },
          ...(s.best ? [{ label: tr(bi(`best line for ${LOSS_NAME[s.loss]}`, `بهترین خط برای ${LOSS_NAME[s.loss]}`)), color: 'g', dash: true }] : []),
        ]}
        ariaLabel="temperature data, prediction line and errors"
      />
      <div className="row">
        <label className="small row" style={{ gap: 4 }}>
          <input type="checkbox" checked={s.outlier} onChange={(e) => set({ outlier: e.target.checked, focus: 0 })} data-testid="toggle-outlier" />
          {tr(bi(`Add a broken-sensor reading (x = ${OUTLIER.x}, y = ${OUTLIER.y})`, `افزودن یک خوانش سنسور خراب (x = ${OUTLIER.x} و y = ${OUTLIER.y})`))}
        </label>
        <label className="small row" style={{ gap: 4 }}>
          <input type="checkbox" checked={s.best} onChange={(e) => set({ best: e.target.checked })} data-testid="toggle-best" />
          {tr(bi('Show the best line for this loss', 'نمایش بهترین خط برای این تابع خطا'))}
        </label>
      </div>
      <p className="small muted">{tr(bi('Drag on the plot to slide the line up or down (changes b).', 'روی نمودار بکش تا خط بالا و پایین برود (b عوض می‌شود).'))}</p>
    </>
  );
}

function BceCenter() {
  const { tr } = useT();
  const [s, set] = useS();
  const d = s.days[s.day];
  const setDay = (p: Partial<Day>) => set({ days: s.days.map((q, i) => (i === s.day ? { ...q, ...p } : q)) });
  const layers: PlotLayer[] = [
    { kind: 'fn', f: (p) => bceL({ p, y: 1 }), color: 'L', width: d.y === 1 ? 2.8 : 1.2, opacity: d.y === 1 ? 1 : 0.3, domain: [0.002, 0.998] },
    { kind: 'fn', f: (p) => bceL({ p, y: 0 }), color: 'L', width: d.y === 0 ? 2.8 : 1.2, opacity: d.y === 0 ? 1 : 0.3, domain: [0.002, 0.998] },
    { kind: 'fn', f: (p) => lossSample('mse', [p], [d.y]), color: 'var(--muted)', dash: '5 4', width: 1.4 },
    { kind: 'vline', at: d.y, color: 'y', label: `y = ${d.y}` },
    { kind: 'line', p: [d.p, bceL(d)], slope: bceG(d), color: 'g', width: 1.5 },
    { kind: 'points', color: 'L', pts: [{ x: d.p, y: bceL(d), label: `BCE = ${fmt(bceL(d), 3)}` }] },
  ];
  return (
    <>
      <div className="row">
        <span className="small muted">{tr(bi('Day:', 'روز:'))}</span>
        {s.days.map((q, i) => (
          <button key={i} className={`btn sm ${i === s.day ? 'active' : ''}`} onClick={() => set({ day: i })}>
            {i + 1} {q.y ? '☂' : '☀'}
          </button>
        ))}
      </div>
      <Plot2D
        testId="bce-plot"
        xDomain={[0, 1]}
        yDomain={[0, 5]}
        layers={layers}
        height={330}
        xLabel={tr(bi('predicted probability of rain p', 'احتمال پیش‌بینی‌شده باران p'))}
        yLabel={tr(bi('loss for this day', 'خطای این روز'))}
        handle={{ x: d.p, y: bceL(d), color: 'L' }}
        onDrag={(p) => setDay({ p: clamp(Math.round(p * 100) / 100, 0.01, 0.99) })}
        legend={[
          { label: tr(bi('BCE if it rains (y = 1): −log p', 'BCE اگر باران بیاید (y = 1): −log p')), color: 'L' },
          { label: tr(bi('squared error (p − y)², for comparison', 'مربع خطا (p − y)² برای مقایسه')), color: 'var(--muted)', dash: true },
          { label: tr(bi('slope = gradient', 'شیب = گرادیان')), color: 'g' },
        ]}
        ariaLabel="binary cross-entropy against predicted probability"
      />
      <Callout tone={Math.abs(d.p - d.y) > 0.7 ? 'warn' : 'info'}>
        {Math.abs(d.p - d.y) > 0.7
          ? tr(bi('Confident and wrong: the loss shoots up. Squared error would stay below 1 here.', 'مطمئن و غلط: خطا به‌شدت بالا می‌رود. مربع خطا اینجا زیر ۱ می‌ماند.'))
          : tr(bi('Drag p toward the wrong answer and watch BCE grow without limit.', 'p را به سمت جواب غلط بکش و ببین BCE بی‌حد بزرگ می‌شود.'))}
      </Callout>
    </>
  );
}

function CceCenter() {
  const { tr } = useT();
  const [s, set] = useS();
  const c = useCce();
  const pt = c.p[s.cls];
  return (
    <>
      <div className="stack" style={{ gap: 6 }} data-testid="cce-bars">
        {CLASSES.map((name, k) => (
          <div key={k} style={{ display: 'grid', gridTemplateColumns: '6em 1fr 4em', gap: 8, alignItems: 'center', cursor: 'pointer' }} onClick={() => set({ cls: k })}>
            <span className={k === s.cls ? 'c-y' : ''} style={{ fontWeight: k === s.cls ? 700 : 400 }}>
              {tr(name)} {k === s.cls ? '✓' : ''}
            </span>
            <div dir="ltr" style={{ height: 20, background: 'var(--surface-2)', borderRadius: 4, outline: k === s.cls ? '2px solid var(--c-y)' : undefined }}>
              <div style={{ width: `${c.p[k] * 100}%`, height: '100%', borderRadius: 4, background: 'var(--c-yhat)' }} />
            </div>
            <span className="num">{fmt(c.p[k], 3)}</span>
          </div>
        ))}
      </div>
      <Plot2D
        testId="cce-plot"
        xDomain={[0, 1]}
        yDomain={[0, 5]}
        height={260}
        xLabel={tr(bi('probability given to the true class', 'احتمالی که به کلاس درست داده شده'))}
        yLabel="CCE = −log p"
        layers={[
          { kind: 'fn', f: (p) => -Math.log(p), color: 'L', width: 2.6, domain: [0.005, 1] },
          { kind: 'line', p: [pt, c.loss], slope: c.dp[s.cls], color: 'g', width: 1.5 },
          { kind: 'points', color: 'L', pts: [{ x: pt, y: c.loss, label: `CCE = ${fmt(c.loss, 3)}` }] },
        ]}
        ariaLabel="categorical cross-entropy against the probability of the true class"
      />
      <p className="small muted">{tr(bi('Click a class to make it the true one. The scores are on the right.', 'روی یک کلاس بزن تا کلاس درست شود. امتیازها در سمت راست هستند.'))}</p>
    </>
  );
}

function Center() {
  const { tr } = useT();
  const [s, set] = useS();
  return (
    <div className="viz">
      <Segmented
        label="stage"
        value={s.stage}
        onChange={(stage) => set({ stage })}
        options={[
          { value: 'reg', label: tr(bi('Regression: MSE, MAE, Huber', 'رگرسیون: MSE، MAE، Huber')) },
          { value: 'bce', label: tr(bi('Rain or not: BCE', 'باران یا نه: BCE')) },
          { value: 'cce', label: tr(bi('3 weather classes: CCE', 'سه نوع هوا: CCE')) },
        ]}
      />
      {s.stage === 'reg' ? <RegCenter /> : s.stage === 'bce' ? <BceCenter /> : <CceCenter />}
    </div>
  );
}

/** What the sign of a parameter gradient says about the direction to move that parameter. */
function dirText(g: number, p: string): Bi {
  const d = `\\partial L/\\partial ${p}`;
  if (Math.abs(g) < 1e-6)
    return bi(`$${d} \\approx 0$: a small change of $${p}$ barely changes the loss.`, `$${d} \\approx 0$: تغییر کوچک $${p}$ تقریباً اثری روی خطا ندارد.`);
  return g < 0
    ? bi(`$${d} < 0$: increasing $${p}$ lowers the loss.`, `$${d} < 0$: زیاد کردن $${p}$ خطا را کم می‌کند.`)
    : bi(`$${d} > 0$: decreasing $${p}$ lowers the loss.`, `$${d} > 0$: کم کردن $${p}$ خطا را کم می‌کند.`);
}

function RegRight() {
  const { tr } = useT();
  const [s, set] = useS();
  const r = useReg();
  const setParam = useLab((st) => st.setParam);
  const terms = r.rows.map((q) => tex(q.l, 4)).join(' + ');
  return (
    <div className="stack">
      <Segmented
        label="loss"
        value={s.loss}
        onChange={(loss) => set({ loss })}
        options={(['mse', 'mae', 'huber'] as RegLoss[]).map((v) => ({ value: v, label: LOSS_NAME[v] }))}
      />
      {s.loss === 'huber' && <Slider label="δ" value={s.delta} min={0.05} max={2} step={0.05} onChange={(delta) => set({ delta })} />}
      <Slider label="w" value={r.w} min={-1} max={3} step={0.01} color="w" onChange={(v) => setParam(W_REF, v)} testId="l06-w" />
      <Slider label="b" value={r.b} min={-5} max={2} step={0.01} color="b" onChange={(v) => setParam(B_REF, v)} testId="l06-b" />
      <Tex display>{`\\cyhat{\\hat y_n} = \\cw{${tex(r.w, 3)}}\\,\\cx{x_n} + \\cb{${texP(r.b, 3)}}, \\qquad e_n = \\cyhat{\\hat y_n} - \\cy{y_n}`}</Tex>
      <Tex display>{`e = (${r.rows.map((q) => tex(q.e, 3)).join(',\\ ')})`}</Tex>
      {s.loss === 'huber' && (
        <Tex display>{`h_\\delta(e) = \\begin{cases} \\tfrac12 e^2 & |e| \\le \\delta \\\\ \\delta\\,(|e| - \\tfrac12\\delta) & |e| > \\delta \\end{cases}`}</Tex>
      )}
      <Tex display>{`\\cL{\\text{${LOSS_NAME[s.loss]}}} = \\frac{1}{${r.N}}\\big(${r.rows.map((_, n) => TERM[s.loss](n + 1)).join(' + ')}\\big)`}</Tex>
      <Tex display>{`= \\frac{1}{${r.N}}\\big(${terms}\\big) = \\cL{${tex(r.L, 4)}}`}</Tex>
      <h4>{tr(bi('Value and gradient', 'مقدار و گرادیان'))}</h4>
      <div className="stat-grid">
        <Stat k={tr(bi(`${LOSS_NAME[s.loss]} value`, `مقدار ${LOSS_NAME[s.loss]}`))} v={fmt(r.L, 4)} color="L" testId="l06-loss" />
        <Stat k="∂L/∂w" v={fmt(r.gw, 4)} color="g" testId="l06-gw" />
        <Stat k="∂L/∂b" v={fmt(r.gb, 4)} color="g" testId="l06-gb" />
        <Stat k={tr(bi('best possible', 'بهترین ممکن'))} v={fmt(r.best.loss, 4)} />
      </div>
      <P className="small muted">{dirText(r.gw, 'w')}</P>
      <P className="small muted">{dirText(r.gb, 'b')}</P>
      <button
        className="btn sm"
        onClick={() => {
          setParam(W_REF, r.best.w);
          setParam(B_REF, r.best.b);
        }}
        data-testid="jump-best"
      >
        {tr(bi('Jump to the best line', 'برو روی بهترین خط'))}
      </button>
    </div>
  );
}

function BceRight() {
  const { tr } = useT();
  const [s, set] = useS();
  const d = s.days[s.day];
  const setDay = (p: Partial<Day>) => set({ days: s.days.map((q, i) => (i === s.day ? { ...q, ...p } : q)) });
  const logTerm = d.y === 1 ? `-\\log(${tex(d.p, 3)})` : `-\\log(1 - ${tex(d.p, 3)})`;
  return (
    <div className="stack">
      <h4>{tr(bi(`Day ${s.day + 1}`, `روز ${s.day + 1}`))}</h4>
      <Segmented
        label="label"
        value={d.y}
        onChange={(y) => setDay({ y })}
        options={[
          { value: 1 as const, label: tr(bi('It rained (y = 1)', 'باران آمد (y = 1)')) },
          { value: 0 as const, label: tr(bi('No rain (y = 0)', 'باران نیامد (y = 0)')) },
        ]}
      />
      <Slider label="p" value={d.p} min={0.01} max={0.99} step={0.01} color="yhat" onChange={(p) => setDay({ p })} testId="l06-p" />
      <Tex display>{'\\cL{\\ell} = -\\big[\\cy{y}\\log \\cyhat{p} + (1-\\cy{y})\\log(1-\\cyhat{p})\\big]'}</Tex>
      <Tex display>{`\\cL{\\ell} = ${logTerm} = \\cL{${tex(bceL(d), 4)}}`}</Tex>
      <Tex display>{`\\cg{\\frac{\\partial \\ell}{\\partial p}} = -\\frac{y}{p} + \\frac{1-y}{1-p} = \\cg{${tex(bceG(d), 4)}}`}</Tex>
      <div className="stat-grid">
        <Stat k={tr(bi('loss value', 'مقدار خطا'))} v={fmt(bceL(d), 4)} color="L" testId="l06-bce" />
        <Stat k={tr(bi('gradient ∂ℓ/∂p', 'گرادیان ∂ℓ/∂p'))} v={fmt(bceG(d), 4)} color="g" />
      </div>
      <P className="small muted">
        {bceG(d) < 0
          ? bi('Negative gradient: raising $p$ lowers the loss (it rained, so say "rain" more confidently).', 'گرادیان منفی: بالا بردن $p$ خطا را کم می‌کند (باران آمده، پس با اطمینان بیشتری بگو «باران»).')
          : bi('Positive gradient: lowering $p$ lowers the loss (it stayed dry).', 'گرادیان مثبت: پایین آوردن $p$ خطا را کم می‌کند (هوا خشک ماند).')}
      </P>
    </div>
  );
}

function CceRight() {
  const { tr } = useT();
  const [s, set] = useS();
  const c = useCce();
  const vec = (v: number[], cmd: string) => `\\begin{bmatrix}${v.map((q) => `\\${cmd}{${tex(q, 3)}}`).join(' & ')}\\end{bmatrix}`;
  return (
    <div className="stack">
      {CLASSES.map((name, k) => (
        <Slider
          key={k}
          label={`z · ${tr(name)}`}
          value={s.logits[k]}
          min={-3}
          max={4}
          step={0.1}
          color="z"
          onChange={(v) => set({ logits: s.logits.map((q, i) => (i === k ? v : q)) })}
          testId={`l06-z${k}`}
        />
      ))}
      <Segmented label="true class" value={s.cls} onChange={(cls) => set({ cls })} options={CLASSES.map((name, k) => ({ value: k, label: tr(name) }))} />
      <Tex display>{`\\cyhat{p_k} = \\frac{e^{\\cz{z_k}}}{\\sum_j e^{\\cz{z_j}}}, \\quad \\cyhat{\\mathbf{p}} = ${vec(c.p, 'cyhat')}`}</Tex>
      <Tex display>{`\\cL{\\ell} = -\\sum_k \\cy{y_k}\\log \\cyhat{p_k} = -\\log(${tex(c.p[s.cls], 3)}) = \\cL{${tex(c.loss, 4)}}`}</Tex>
      <Tex display>{`\\cg{\\frac{\\partial \\ell}{\\partial \\mathbf{p}}} = ${vec(c.dp, 'cg')}`}</Tex>
      <Tex display>{`\\cg{\\frac{\\partial \\ell}{\\partial \\mathbf{z}}} = \\cyhat{\\mathbf{p}} - \\cy{\\mathbf{y}} = ${vec(c.dz, 'cg')}`}</Tex>
      <div className="stat-grid">
        <Stat k={tr(bi('loss value', 'مقدار خطا'))} v={fmt(c.loss, 4)} color="L" testId="l06-cce" />
        <Stat k={tr(bi('p of true class', 'p کلاس درست'))} v={fmt(c.p[s.cls], 4)} color="yhat" />
      </div>
      <P className="small muted">
        {bi(
          'The gradient with respect to the scores is negative only for the true class: raise its score, lower the others in proportion to their probability.',
          'گرادیان نسبت به امتیازها فقط برای کلاس درست منفی است: امتیاز آن را بالا ببر و بقیه را به نسبت احتمالشان پایین بیاور.',
        )}
      </P>
    </div>
  );
}

function Right() {
  const [s] = useS();
  return s.stage === 'reg' ? <RegRight /> : s.stage === 'bce' ? <BceRight /> : <CceRight />;
}

function RegBottom() {
  const { tr } = useT();
  const [s, set] = useS();
  const r = useReg();
  const q = r.rows[Math.min(s.focus, r.N - 1)];
  const o = { delta: s.delta };
  const per = (name: RegLoss) => (yh: number) => lossSample(name, [yh], [q.y], o);
  return (
    <div className="viz-grid">
      <table className="tbl" data-testid="loss-table">
        <thead>
          <tr>
            <th>n</th>
            <th className="num c-x">x</th>
            <th className="num c-y">y</th>
            <th className="num c-yhat">ŷ</th>
            <th className="num c-L">e = ŷ − y</th>
            <th className="num">e²</th>
            {s.loss !== 'mse' && <th className="num">{s.loss === 'mae' ? '|e|' : 'h_δ(e)'}</th>}
            <th className="num c-g">∂ℓₙ/∂ŷ</th>
          </tr>
        </thead>
        <tbody>
          {r.rows.map((row, n) => (
            <tr key={n} className={n === s.focus ? 'hl' : ''} onClick={() => set({ focus: n })} style={{ cursor: 'pointer' }}>
              <td>
                {n + 1}
                {row.outlier ? ' ⚠' : ''}
              </td>
              <td className="num">{fmt(row.x, 2)}</td>
              <td className="num">{fmt(row.y, 2)}</td>
              <td className="num">{fmt(row.yhat, 3)}</td>
              <td className="num">{fmtSigned(row.e, 3)}</td>
              <td className="num">{fmt(row.e * row.e, 4)}</td>
              {s.loss !== 'mse' && <td className="num">{fmt(row.l, 4)}</td>}
              <td className="num">{fmtSigned(row.g, 3)}</td>
            </tr>
          ))}
          <tr className="hl">
            <td colSpan={5}>
              <strong>{tr(bi(`mean = ${LOSS_NAME[s.loss]}`, `میانگین = ${LOSS_NAME[s.loss]}`))}</strong>
            </td>
            <td className="num">{s.loss === 'mse' ? <strong className="c-L">{fmt(r.L, 4)}</strong> : fmt(r.rows.reduce((a, row) => a + row.e * row.e, 0) / r.N, 4)}</td>
            {s.loss !== 'mse' && (
              <td className="num">
                <strong className="c-L">{fmt(r.L, 4)}</strong>
              </td>
            )}
            <td />
          </tr>
        </tbody>
      </table>
      <div>
        <div className="small muted">
          {tr(bi(`Loss of example ${s.focus + 1} as its prediction ŷ varies (target y = ${fmt(q.y, 2)})`, `خطای مثال ${s.focus + 1} وقتی پیش‌بینی‌اش ŷ تغییر می‌کند (هدف y = ${fmt(q.y, 2)})`))}
        </div>
        <Plot2D
          xDomain={[q.y - 2.5, q.y + 2.5]}
          yDomain={[-0.1, 3]}
          height={200}
          xLabel="ŷ"
          yLabel="ℓ(ŷ)"
          layers={[
            ...(['mse', 'mae', 'huber'] as RegLoss[]).map(
              (name): PlotLayer => ({ kind: 'fn', f: per(name), color: name === s.loss ? 'L' : 'var(--muted)', width: name === s.loss ? 2.6 : 1.2, opacity: name === s.loss ? 1 : 0.6 }),
            ),
            { kind: 'vline', at: q.y, color: 'y', label: 'y' },
            { kind: 'line', p: [q.yhat, q.l], slope: q.g, color: 'g', width: 1.5 },
            { kind: 'points', color: 'L', pts: [{ x: q.yhat, y: q.l, label: `ℓ = ${fmt(q.l, 3)}, ∂ℓ/∂ŷ = ${fmt(q.g, 3)}` }] },
          ]}
          legend={(['mse', 'mae', 'huber'] as RegLoss[]).map((name) => ({ label: LOSS_NAME[name], color: name === s.loss ? 'L' : 'var(--muted)' }))}
        />
      </div>
    </div>
  );
}

function BceBottom() {
  const { tr } = useT();
  const [s, set] = useS();
  const mean = s.days.reduce((a, d) => a + bceL(d), 0) / s.days.length;
  return (
    <table className="tbl" data-testid="bce-table">
      <thead>
        <tr>
          <th>{tr(bi('day', 'روز'))}</th>
          <th className="num c-yhat">p</th>
          <th className="num c-y">y</th>
          <th className="num c-L">BCE</th>
          <th className="num c-g">∂ℓ/∂p</th>
          <th className="num">(p − y)²</th>
        </tr>
      </thead>
      <tbody>
        {s.days.map((d, i) => (
          <tr key={i} className={i === s.day ? 'hl' : ''} onClick={() => set({ day: i })} style={{ cursor: 'pointer' }}>
            <td>
              {i + 1} {d.y ? tr(bi('rain', 'باران')) : tr(bi('dry', 'خشک'))}
            </td>
            <td className="num">{fmt(d.p, 2)}</td>
            <td className="num">{d.y}</td>
            <td className="num">{fmt(bceL(d), 4)}</td>
            <td className="num">{fmtSigned(bceG(d), 3)}</td>
            <td className="num">{fmt(lossSample('mse', [d.p], [d.y]), 4)}</td>
          </tr>
        ))}
        <tr className="hl">
          <td colSpan={3}>
            <strong>{tr(bi('mean over the days = batch BCE', 'میانگین روزها = BCE دسته'))}</strong>
          </td>
          <td className="num c-L" data-testid="l06-bce-mean">
            <strong>{fmt(mean, 4)}</strong>
          </td>
          <td colSpan={2} />
        </tr>
      </tbody>
    </table>
  );
}

function CceBottom() {
  const { tr } = useT();
  const [s] = useS();
  const c = useCce();
  return (
    <table className="tbl" data-testid="cce-table">
      <thead>
        <tr>
          <th>{tr(bi('class', 'کلاس'))}</th>
          <th className="num c-z">z</th>
          <th className="num">eᶻ</th>
          <th className="num c-yhat">p</th>
          <th className="num c-y">y</th>
          <th className="num c-L">−y·log p</th>
          <th className="num c-g">∂ℓ/∂p</th>
          <th className="num c-g">∂ℓ/∂z</th>
        </tr>
      </thead>
      <tbody>
        {CLASSES.map((name, k) => (
          <tr key={k} className={k === s.cls ? 'hl' : ''}>
            <td>{tr(name)}</td>
            <td className="num">{fmt(s.logits[k], 2)}</td>
            <td className="num">{fmt(Math.exp(s.logits[k]), 3)}</td>
            <td className="num">{fmt(c.p[k], 4)}</td>
            <td className="num">{c.y[k]}</td>
            <td className="num">{fmt(c.y[k] ? -Math.log(c.p[k]) : 0, 4)}</td>
            <td className="num">{fmtSigned(c.dp[k], 3)}</td>
            <td className="num">{fmtSigned(c.dz[k], 4)}</td>
          </tr>
        ))}
        <tr className="hl">
          <td colSpan={5}>
            <strong>{tr(bi('sum over classes = CCE', 'جمع روی کلاس‌ها = CCE'))}</strong>
          </td>
          <td className="num c-L">
            <strong>{fmt(c.loss, 4)}</strong>
          </td>
          <td colSpan={2} />
        </tr>
      </tbody>
    </table>
  );
}

function Bottom() {
  const [s] = useS();
  return s.stage === 'reg' ? <RegBottom /> : s.stage === 'bce' ? <BceBottom /> : <CceBottom />;
}

export const lesson06: LessonDef = {
  id: ID,
  title: bi('Predictions and Loss', 'پیش‌بینی و تابع خطا'),
  short: bi('Loss', 'تابع خطا'),
  objectives: [
    bi('Compute each example\'s error and squared error, then average them into MSE', 'خطا و مربع خطای هر مثال را حساب کنی و از میانگین آنها MSE را بسازی'),
    bi('Lower the MSE by hand with w and b, guided by the gradient', 'با w و b و به کمک گرادیان، MSE را با دست کم کنی'),
    bi('Tell a loss value from a loss gradient', 'مقدار خطا را از گرادیان خطا تشخیص بدهی'),
    bi('Choose between MSE, MAE, Huber, BCE and CCE for a task', 'برای یک مسئله بین MSE، MAE، Huber، BCE و CCE انتخاب کنی'),
  ],
  presetId: 'temperature',
  revealSteps: 7,
  Explain,
  Center,
  Right,
  Bottom,
  checkpoint: {
    question: bi('A model says "99 % chance of rain" ($p = 0.99$) and it stays dry ($y = 0$). What is its BCE for that day?', 'مدلی می‌گوید «۹۹ درصد احتمال باران» ($p = 0.99$) و هوا خشک می‌ماند ($y = 0$). BCE آن روز چقدر است؟'),
    choices: [
      { id: 'a', label: bi('About 4.6: confident and wrong is punished hard', 'حدود ۴.۶: اطمینانِ غلط به‌شدت جریمه می‌شود') },
      { id: 'b', label: bi('0.99', '۰.۹۹') },
      { id: 'c', label: bi('About 0.01, because 0.99 is close to 1', 'حدود ۰.۰۱، چون ۰.۹۹ به ۱ نزدیک است') },
    ],
    answer: 'a',
    explain: bi(
      'With $y = 0$ the loss is $-\\log(1 - p) = -\\log(0.01) \\approx 4.6$. Set a dry day to $p = 0.99$ in the rain tab to check.',
      'با $y = 0$ خطا برابر $-\\log(1 - p) = -\\log(0.01) \\approx 4.6$ است. در زبانه باران یک روز خشک را روی $p = 0.99$ بگذار و بررسی کن.',
    ),
  },
};
