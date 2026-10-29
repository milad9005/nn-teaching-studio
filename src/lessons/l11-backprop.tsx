// Lesson 11: backpropagation on the branching network (1 → 2 → 2 → 1, ReLU, ReLU, linear).
// Shared lab store (preset "branching"): the stepper drives four phases — forward, loss,
// backward, optimizer update — and the canvas, the narration, the inspector, the parameter
// table and the gradient check all read the same engine results.

import type { ReactNode } from 'react';
import type { LessonDef } from './types';
import { bi, useT } from '../i18n';
import { useLab, useLabDerived, visibility, type Selection } from '../state/lab';
import { useLessonState } from '../state/lessonState';
import { NetworkCanvas } from '../components/NetworkCanvas';
import { Inspector } from '../components/Inspector';
import { GradCheckPanel } from '../components/LabPanels';
import { Tex } from '../components/Tex';
import { Slider, Stat } from '../components/controls';
import { Advanced, Callout, Cycle, P, Predict, Reveal } from '../components/teaching';
import { actDeriv, datasetLoss, fmt, getGrad, getParam, paramRefs, sizes, tex, texP, type ActivationName, type ParamRef } from '../engine';
import { gradientCost, pathCount, phaseOf, type Phase } from './math/l11';

const ID = 'l11';

interface S {
  /** What-if network for the cost comparison: hidden width and number of hidden layers. */
  width: number;
  depth: number;
}
const INIT: S = { width: 64, depth: 3 };
const useS = () => useLessonState<S>(ID, INIT);

const FN: Record<ActivationName, string> = { linear: '', relu: '\\relu', sigmoid: '\\sigma', tanh: '\\tanh', softmax: '\\operatorname{softmax}' };
const vec = (a: number[]) => `\\left(${a.map((v) => tex(v, 3)).join(',\\ ')}\\right)`;
const big = (n: number) => (n < 1e12 ? Math.round(n).toLocaleString('en-US') : n.toExponential(2));

function Explain() {
  const { tr } = useT();
  return (
    <>
      <Cycle active={3} />
      <Reveal at={1}>
        <h3>{tr(bi('1. One training step, four phases', '۱. یک گام آموزش، چهار مرحله'))}</h3>
        <P>
          {bi(
            'A network is a computational graph: every value is computed from the values before it. One training step walks that graph twice and then changes the weights:',
            'شبکه یک گراف محاسباتی (computational graph) است: هر مقدار از مقدارهای قبل از خودش ساخته می‌شود. یک گام آموزش دو بار این گراف را طی می‌کند و بعد وزن‌ها را تغییر می‌دهد:',
          )}
        </P>
        <ol className="small">
          <li>{tr(bi('Forward: compute every z and a, layer by layer, left to right.', 'رو به جلو (Forward): همه z و aها را لایه به لایه از چپ به راست حساب کن.'))}</li>
          <li>{tr(bi('Loss: compare the output ŷ with the target y.', 'خطا (Loss): خروجی ŷ را با هدف y مقایسه کن.'))}</li>
          <li>{tr(bi('Backward: compute ∂L/∂(everything), right to left.', 'رو به عقب (Backward): ‎∂L/∂(همه‌چیز) را از راست به چپ حساب کن.'))}</li>
          <li>{tr(bi('Update: the optimizer moves every weight a little against its gradient.', 'به‌روزرسانی (Update): بهینه‌ساز هر وزن را کمی خلاف جهت گرادیانش جابه‌جا می‌کند.'))}</li>
        </ol>
      </Reveal>
      <Reveal at={2}>
        <h3>{tr(bi('2. Forward and loss', '۲. رو به جلو و خطا'))}</h3>
        <P>
          {bi(
            'Press "Forward step" a few times. Each layer only needs the previous layer\'s activations. With $\\cx{x} = 1$ the output is $\\cyhat{\\hat y} = 2.5$; the target is $\\cy{y} = 1$, so $\\cL{L} = (2.5 - 1)^2 = 2.25$. Every $z$ and $a$ is kept in memory: the backward pass will reuse them.',
            'چند بار «گام رو به جلو» را بزن. هر لایه فقط به فعال‌سازی‌های (activation) لایه قبل نیاز دارد. با $\\cx{x} = 1$ خروجی $\\cyhat{\\hat y} = 2.5$ است؛ هدف $\\cy{y} = 1$ است، پس $\\cL{L} = (2.5 - 1)^2 = 2.25$. همه $z$ و $a$ها در حافظه نگه داشته می‌شوند: گذر رو به عقب از آنها دوباره استفاده می‌کند.',
          )}
        </P>
      </Reveal>
      <Reveal at={3}>
        <Predict
          id={ID + '-dir'}
          question={bi(
            'The output (2.5) is above the target (1). If the first weight $W_1[0,0]$ (from $x$ to the top hidden neuron) grows a little, does the loss go up or down?',
            'خروجی (۲٫۵) بالاتر از هدف (۱) است. اگر وزن اول $W_1[0,0]$ (از $x$ به نورون بالایی لایه پنهان) کمی بزرگ شود، خطا بالا می‌رود یا پایین؟',
          )}
          choices={[
            { id: 'up', label: bi('Up', 'بالا') },
            { id: 'down', label: bi('Down', 'پایین') },
            { id: 'same', label: bi('No change', 'تغییری نمی‌کند') },
          ]}
          answer="up"
          explain={bi(
            'Both routes from that weight to the output have only positive weights, so a bigger $W_1[0,0]$ makes $\\hat y$ even bigger and the loss grows: $\\partial L/\\partial W_1[0,0] = +6$. Click that edge in the network to see why.',
            'هر دو مسیر از این وزن تا خروجی فقط وزن‌های مثبت دارند، پس $W_1[0,0]$ بزرگ‌تر، $\\hat y$ را باز هم بزرگ‌تر می‌کند و خطا زیاد می‌شود: $\\partial L/\\partial W_1[0,0] = +6$. روی همین یال در شبکه کلیک کن تا دلیلش را ببینی.',
          )}
        />
      </Reveal>
      <Reveal at={4}>
        <h3>{tr(bi('3. Backward: one layer at a time', '۳. رو به عقب: هر بار یک لایه'))}</h3>
        <P>
          {bi(
            'Start at the loss: $\\cg{\\partial L/\\partial \\hat y} = 2(\\hat y - y) = 3$. Then step left. For each neuron, its $\\cg{\\partial L/\\partial z}$ is "how much the loss cares about this neuron". It is computed once, from the layer to its right, and then reused twice: for the gradients of all weights entering the neuron ($\\partial L/\\partial z \\times$ input) and to pass the signal further left.',
            'از خطا شروع کن: $\\cg{\\partial L/\\partial \\hat y} = 2(\\hat y - y) = 3$. بعد یک قدم به چپ برو. برای هر نورون، $\\cg{\\partial L/\\partial z}$ یعنی «خطا چقدر به این نورون اهمیت می‌دهد». این مقدار یک بار، از روی لایه سمت راستش حساب می‌شود و بعد دو بار دوباره استفاده می‌شود: برای گرادیان همه وزن‌هایی که وارد نورون می‌شوند ($\\partial L/\\partial z \\times$ ورودی) و برای فرستادن سیگنال به لایه‌های چپ‌تر.',
          )}
        </P>
      </Reveal>
      <Reveal at={5}>
        <h3>{tr(bi('4. Paths that meet are added', '۴. مسیرهایی که به هم می‌رسند جمع می‌شوند'))}</h3>
        <P>
          {bi(
            '$W_1[0,0]$ feeds the top hidden neuron, which feeds both neurons of the next layer. So a change in it reaches the loss along two paths. Along each path the local derivatives multiply (chain rule); here each path contributes $3$. The total effect is the sum: $3 + 3 = 6$.',
            '$W_1[0,0]$ به نورون بالایی لایه پنهان می‌رود که خودش به هر دو نورون لایه بعد وصل است. پس تغییر آن از دو مسیر به خطا می‌رسد. در طول هر مسیر، مشتق‌های محلی در هم ضرب می‌شوند (قاعده زنجیره‌ای)؛ اینجا سهم هر مسیر $3$ است. اثر کل، جمع آنهاست: $3 + 3 = 6$.',
          )}
        </P>
        <Callout>
          {tr(
            bi(
              'Click the edge W1[0,0] (or use the button on the right). The highlighted paths and the path list in the inspector show each product and their sum. Backprop never lists paths: summing ∂L/∂z over the next layer already adds them.',
              'روی یال W1[0,0] کلیک کن (یا دکمه سمت راست را بزن). مسیرهای پررنگ و فهرست مسیرها در بازرس (inspector) هر حاصل‌ضرب و جمعشان را نشان می‌دهند. Backprop هیچ‌وقت مسیرها را تک‌تک نمی‌شمارد: جمع ∂L/∂z روی لایه بعد، خودش آنها را جمع کرده است.',
            ),
          )}
        </Callout>
      </Reveal>
      <Reveal at={6}>
        <h3>{tr(bi('5. Why backprop is efficient', '۵. چرا Backprop کارآمد است'))}</h3>
        <P>
          {bi(
            'The obvious alternative is to nudge each parameter and re-run the network: at least one extra forward pass per parameter. This tiny net has 13 parameters; real ones have millions to billions. Backprop gets all gradients from one forward pass plus one backward pass of similar cost, no matter how many parameters there are.',
            'راه ساده‌تر این است که هر پارامتر را تکان بدهیم و شبکه را دوباره اجرا کنیم: دست‌کم یک گذر رو به جلوی اضافه به ازای هر پارامتر. این شبکه کوچک ۱۳ پارامتر دارد؛ شبکه‌های واقعی میلیون‌ها تا میلیاردها. Backprop همه گرادیان‌ها را با یک گذر رو به جلو و یک گذر رو به عقب با هزینه‌ای مشابه به دست می‌آورد، هر چقدر هم پارامترها زیاد باشند.',
          )}
        </P>
        <P>
          {bi(
            'The slow way is still useful as a test: the gradient check in the bottom panel compares backprop with finite differences. Finally, "Optimizer step" uses all gradients at once: $\\theta \\leftarrow \\theta - \\clr{\\eta}\\,\\cg{\\nabla L}$.',
            'راه کند هنوز به عنوان آزمون مفید است: بررسی گرادیان (gradient check) در پنل پایین Backprop را با تفاضل متناهی مقایسه می‌کند. در آخر «گام بهینه‌ساز» همه گرادیان‌ها را یک‌جا استفاده می‌کند: $\\theta \\leftarrow \\theta - \\clr{\\eta}\\,\\cg{\\nabla L}$.',
          )}
        </P>
        <Advanced title={bi('Backprop in matrix form', 'Backprop به شکل ماتریسی')}>
          <Tex display>{"\\delta^{(L)} = \\frac{\\partial L}{\\partial \\hat y}\\odot f_L'(z^{(L)}),\\qquad \\delta^{(l)} = \\big(W^{(l+1)\\top}\\delta^{(l+1)}\\big)\\odot f_l'(z^{(l)})"}</Tex>
          <Tex display>{'\\frac{\\partial L}{\\partial W^{(l)}} = \\delta^{(l)}\\,a^{(l-1)\\top},\\qquad \\frac{\\partial L}{\\partial b^{(l)}} = \\delta^{(l)}'}</Tex>
          <P>
            {bi(
              'The product $W^{\\top}\\delta$ is exactly the "sum over next-layer neurons" — the addition of converging paths. Each layer costs two matrix products (one for $\\partial L/\\partial W$, one for $W^\\top\\delta$), so the backward pass costs about twice the forward pass. The price is memory: all activations from the forward pass must be stored. This is reverse-mode automatic differentiation, which PyTorch and JAX run for you.',
              'حاصل‌ضرب $W^{\\top}\\delta$ دقیقاً همان «جمع روی نورون‌های لایه بعد» است — یعنی جمع مسیرهایی که به هم می‌رسند. هر لایه دو ضرب ماتریسی هزینه دارد (یکی برای $\\partial L/\\partial W$ و یکی برای $W^\\top\\delta$)، پس گذر رو به عقب حدود دو برابر گذر رو به جلو هزینه دارد. بهایش حافظه است: همه فعال‌سازی‌های گذر رو به جلو باید ذخیره شوند. این همان مشتق‌گیری خودکار در حالت معکوس (reverse-mode autodiff) است که PyTorch و JAX برایت اجرا می‌کنند.',
            )}
          </P>
        </Advanced>
      </Reveal>
    </>
  );
}

const PHASES: { id: Exclude<Phase, 'idle'>; label: ReturnType<typeof bi>; desc: ReturnType<typeof bi> }[] = [
  { id: 'forward', label: bi('1 · Forward', '۱ · رو به جلو'), desc: bi('z and a, left → right', 'z و a، چپ ← راست') },
  { id: 'loss', label: bi('2 · Loss', '۲ · خطا'), desc: bi('compare ŷ with y', 'مقایسه ŷ با y') },
  { id: 'backward', label: bi('3 · Backward', '۳ · رو به عقب'), desc: bi('∂L/∂z, right → left', '∂L/∂z، راست ← چپ') },
  { id: 'update', label: bi('4 · Update', '۴ · به‌روزرسانی'), desc: bi('θ ← θ − η·∇L', 'θ ← θ − η·∇L') },
];

function PhaseControls() {
  const { tr, t } = useT();
  const stepper = useLab((s) => s.stepper);
  const running = useLab((s) => s.running);
  const L = useLab((s) => s.net.layers.length);
  const forwardStep = useLab((s) => s.forwardStep);
  const backwardStep = useLab((s) => s.backwardStep);
  const optimizerStep = useLab((s) => s.optimizerStep);
  const showAll = useLab((s) => s.showAll);
  const resetTraining = useLab((s) => s.resetTraining);
  const error = useLab((s) => s.error);
  const phase = phaseOf(stepper, L);
  const toLoss = () => {
    // Run the remaining forward steps up to the loss.
    for (let i = 0; i < L + 3; i++) {
      const st = useLab.getState().stepper;
      if (st && st.phase === 'forward' && st.k >= L + 2) break;
      useLab.getState().forwardStep();
    }
  };
  const order = PHASES.findIndex((p) => p.id === phase);
  return (
    <div className="stack">
      <div className="factor-row" data-testid="l11-phases">
        {PHASES.map((p, i) => (
          <div key={p.id} style={{ display: 'contents' }}>
            {i > 0 && <span className="factor-op">→</span>}
            <div className={`factor ${phase === p.id ? 'on' : ''}`} style={{ cursor: 'default', opacity: order >= 0 && i > order ? 0.55 : 1 }} data-testid={`l11-phase-${p.id}`}>
              <strong>{tr(p.label)}</strong>
              <div className="small muted">{tr(p.desc)}</div>
            </div>
          </div>
        ))}
      </div>
      <div className="row" style={{ justifyContent: 'center' }}>
        <button className="btn" onClick={forwardStep} disabled={running} data-testid="l11-forward">
          → {t('run.forward')}
        </button>
        <button className="btn" onClick={toLoss} disabled={running} data-testid="l11-loss">
          Σ {tr(bi('Compute loss', 'محاسبه خطا'))}
        </button>
        <button className="btn" onClick={backwardStep} disabled={running} data-testid="l11-backward">
          ← {t('run.backward')}
        </button>
        <button className="btn primary" onClick={() => optimizerStep()} disabled={running} data-testid="l11-update">
          ⟳ {t('run.optimizer')}
        </button>
        {stepper && (
          <button className="btn ghost sm" onClick={showAll}>
            {t('run.showAll')}
          </button>
        )}
        <button className="btn ghost sm" onClick={resetTraining} data-testid="l11-reset">
          ↺ {tr(bi('Reset weights', 'بازنشانی وزن‌ها'))}
        </button>
      </div>
      {error && <div className="callout err small">{t(`err.${error}`)}</div>}
    </div>
  );
}

/** What the current step computes, with the engine's numbers. */
function Narration() {
  const { tr } = useT();
  const stepper = useLab((s) => s.stepper);
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const loss = useLab((s) => s.loss);
  const selection = useLab((s) => s.selection);
  const last = useLab((s) => s.lastUpdate);
  const { one } = useLabDerived();
  const L = net.layers.length;
  const trace = one.trace;
  const box = (title: string, body: ReactNode) => (
    <div className="card flat" data-testid="l11-narration">
      <div className="small stepper-label">{title}</div>
      {body}
    </div>
  );

  if (!stepper) {
    return box(
      tr(bi('Everything shown', 'همه‌چیز نمایش داده شده')),
      <P className="small">
        {bi(
          'All forward values and gradients are visible. Use the buttons above to replay one training step phase by phase, or click a weight to see how its gradient is built.',
          'همه مقدارهای رو به جلو و گرادیان‌ها دیده می‌شوند. با دکمه‌های بالا یک گام آموزش را مرحله به مرحله دوباره ببین، یا روی یک وزن کلیک کن تا ببینی گرادیانش چطور ساخته می‌شود.',
        )}
      </P>,
    );
  }

  if (stepper.phase === 'update') {
    if (!last) return box(tr(bi('Update', 'به‌روزرسانی')), null);
    const ref: ParamRef =
      selection?.type === 'weight'
        ? paramRefs(net).find((p) => p.kind === 'W' && p.layer === selection.layer && p.j === selection.j && p.i === selection.i)!
        : selection?.type === 'bias'
          ? paramRefs(net).find((p) => p.kind === 'b' && p.layer === selection.layer && p.j === selection.j)!
          : paramRefs(net)[0];
    const before = getParam(last.before, ref);
    const g = getGrad(last.grads, ref);
    const Lb = datasetLoss(last.before, data, loss);
    const La = datasetLoss(net, data, loss);
    const nP = last.delta.filter((d) => d !== 0).length;
    return box(
      tr(bi('Phase 4 · optimizer update', 'مرحله ۴ · به‌روزرسانی بهینه‌ساز')),
      <>
        <Tex display>{`\\theta_{\\text{new}} = \\theta_{\\text{old}} - \\clr{\\eta}\\,\\cg{\\nabla L},\\qquad \\clr{\\eta} = ${tex(last.lr, 4)}`}</Tex>
        <Tex display>{`\\cw{\\text{${ref.id}}}:\\ ${tex(before, 4)} - ${tex(last.lr, 4)}\\cdot${texP(g, 4)} = ${tex(getParam(net, ref), 4)}`}</Tex>
        <Tex display>{`\\cL{L}:\\ ${tex(Lb, 4)} \\;\\to\\; ${tex(La, 4)}`}</Tex>
        <P className="small muted">
          {bi(
            `All ${nP} parameters moved at once, each against its own gradient (select a weight or bias to see its update). The forward values above are already those of the new weights.`,
            `هر ${nP} پارامتر یک‌جا جابه‌جا شدند، هر کدام خلاف جهت گرادیان خودش (یک وزن یا بایاس را انتخاب کن تا به‌روزرسانی‌اش را ببینی). مقدارهای رو به جلوی بالا حالا مال وزن‌های جدیدند.`,
          )}
        </P>
      </>,
    );
  }

  if (stepper.phase === 'forward') {
    const k = stepper.k;
    if (k === 1)
      return box(
        tr(bi(`Phase 1 · forward 1/${L + 2}: the input`, `مرحله ۱ · رو به جلو 1/${L + 2}: ورودی`)),
        <Tex display>{`\\cx{a^{(0)}} = \\cx{x} = ${vec(trace.a[0])}`}</Tex>,
      );
    if (k >= L + 2) {
      const out = trace.a[L];
      const y = data.Y[0];
      return box(
        tr(bi('Phase 2 · loss', 'مرحله ۲ · خطا')),
        <Tex display>{`\\cL{L} = (\\cyhat{\\hat y} - \\cy{y})^2 = (${tex(out[0], 4)} - ${texP(y[0])})^2 = ${tex(one.loss, 4)}`}</Tex>,
      );
    }
    const l = k - 1; // activation layer just computed (1..L)
    const layer = net.layers[l - 1];
    const prev = trace.a[l - 1];
    return box(
      tr(bi(`Phase 1 · forward ${k}/${L + 2}: layer ${l}`, `مرحله ۱ · رو به جلو ${k}/${L + 2}: لایه ${l}`)),
      <>
        {layer.W.map((row, j) => (
          <Tex key={j} display>
            {`\\cz{z^{(${l})}_{${j + 1}}} = ${row.map((w, i) => `${texP(w)}\\cdot${texP(prev[i])}`).join(' + ')} + ${texP(layer.b[j])} = ${tex(trace.z[l - 1][j], 4)}\\quad\\Rightarrow\\quad \\ca{a^{(${l})}_{${j + 1}}} = ${FN[layer.act]}${layer.act === 'linear' ? '' : '('}${tex(trace.z[l - 1][j], 4)}${layer.act === 'linear' ? '' : ')'} = ${tex(trace.a[l][j], 4)}`}
          </Tex>
        ))}
      </>,
    );
  }

  // Backward.
  const k = stepper.k;
  if (k === 1)
    return box(
      tr(bi(`Phase 3 · backward 1/${L + 1}: start at the loss`, `مرحله ۳ · رو به عقب 1/${L + 1}: شروع از خطا`)),
      <Tex display>{`\\cg{\\frac{\\partial L}{\\partial \\hat y}} = 2(\\cyhat{\\hat y} - \\cy{y}) = 2(${tex(trace.a[L][0], 4)} - ${texP(data.Y[0][0])}) = ${tex(one.dOut[0], 4)}`}</Tex>,
    );
  const li = L - (k - 1); // 0-based weight layer whose dz is revealed
  const layer = net.layers[li];
  const dz = one.back.dz[li];
  const aIn = trace.a[li];
  const isOut = li === L - 1;
  return box(
    tr(bi(`Phase 3 · backward ${k}/${L + 1}: layer ${li + 1}`, `مرحله ۳ · رو به عقب ${k}/${L + 1}: لایه ${li + 1}`)),
    <>
      {dz.map((d, j) => {
        const fprime = actDeriv(layer.act, trace.z[li][j]);
        const incoming = isOut
          ? texP(one.dOut[j], 4)
          : `\\underbrace{\\left(${net.layers[li + 1].W.map((row, q) => `${texP(row[j])}\\cdot${texP(one.back.dz[li + 1][q])}`).join(' + ')}\\right)}_{\\sum_k W_{kj}\\,\\delta_k}`;
        return (
          <Tex key={j} display>
            {`\\cg{\\frac{\\partial L}{\\partial z^{(${li + 1})}_{${j + 1}}}} = ${incoming}\\cdot \\underbrace{${tex(fprime)}}_{f'(z)} = ${tex(d, 4)}`}
          </Tex>
        );
      })}
      <Tex display>{`\\cg{\\frac{\\partial L}{\\partial W^{(${li + 1})}}} = \\delta\\,a^{(${li})\\top} = \\begin{pmatrix}${dz.map((d) => aIn.map((a) => tex(d * a, 3)).join(' & ')).join(' \\\\ ')}\\end{pmatrix},\\quad \\cg{\\frac{\\partial L}{\\partial b^{(${li + 1})}}} = ${vec(dz)}`}</Tex>
      <P className="small muted">
        {li > 0
          ? bi(
              'Each ∂L/∂z above is computed once and reused: for every weight entering the neuron, and to pass the gradient one layer further left.',
              'هر ∂L/∂z بالا یک بار حساب می‌شود و دوباره استفاده می‌شود: برای هر وزنی که وارد نورون می‌شود و برای فرستادن گرادیان یک لایه به چپ‌تر.',
            )
          : bi('First layer reached: every gradient is known. Next: the optimizer update.', 'به لایه اول رسیدیم: همه گرادیان‌ها معلوم‌اند. بعدی: به‌روزرسانی بهینه‌ساز.')}
      </P>
    </>,
  );
}

function Center() {
  return (
    <div className="center-stack">
      <PhaseControls />
      <div className="canvas-wrap" style={{ minHeight: 280 }}>
        <NetworkCanvas />
      </div>
      <Narration />
    </div>
  );
}

function ParamTable() {
  const { tr } = useT();
  const net = useLab((s) => s.net);
  const stepper = useLab((s) => s.stepper);
  const selection = useLab((s) => s.selection);
  const select = useLab((s) => s.select);
  const { one } = useLabDerived();
  const L = net.layers.length;
  const vis = visibility(stepper, L);
  const selOf = (p: ParamRef): Selection => (p.kind === 'W' ? { type: 'weight', layer: p.layer, j: p.j, i: p.i } : { type: 'bias', layer: p.layer, j: p.j });
  const isSel = (p: ParamRef) => JSON.stringify(selOf(p)) === JSON.stringify(selection);
  return (
    <div style={{ maxHeight: 260, overflow: 'auto' }}>
      <table className="tbl" data-testid="l11-params">
        <thead>
          <tr>
            <th>θ</th>
            <th className="num c-w">{tr(bi('value', 'مقدار'))}</th>
            <th className="num c-g">∂L/∂θ</th>
          </tr>
        </thead>
        <tbody>
          {paramRefs(net).map((p) => (
            <tr key={p.id} className={isSel(p) ? 'hl' : ''} onClick={() => select(selOf(p))} style={{ cursor: 'pointer' }}>
              <td className={`mono ${p.kind === 'W' ? 'c-w' : 'c-b'}`}>{p.id}</td>
              <td className="num">{fmt(getParam(net, p), 4)}</td>
              <td className="num c-g">{vis.grad(p.layer + 1) ? fmt(getGrad(one.back.grads, p), 4) : '?'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Right() {
  const { tr } = useT();
  const select = useLab((s) => s.select);
  const quick: { label: string; sel: Selection }[] = [
    { label: 'W1[0,0]', sel: { type: 'weight', layer: 0, j: 0, i: 0 } },
    { label: 'W1[1,0]', sel: { type: 'weight', layer: 0, j: 1, i: 0 } },
    { label: 'b1[0]', sel: { type: 'bias', layer: 0, j: 0 } },
    { label: 'W3[0,1]', sel: { type: 'weight', layer: 2, j: 0, i: 1 } },
  ];
  return (
    <div className="stack">
      <div className="small muted">{tr(bi('Click a weight (edge) or neuron in the network, or pick one:', 'روی یک وزن (یال) یا نورون در شبکه کلیک کن، یا یکی را انتخاب کن:'))}</div>
      <div className="row">
        {quick.map((q) => (
          <button key={q.label} className="btn sm mono" onClick={() => select(q.sel)} data-testid={`l11-pick-${q.label}`}>
            {q.label}
          </button>
        ))}
      </div>
      <Inspector />
      <h4>{tr(bi('All weight and bias gradients', 'گرادیان همه وزن‌ها و بایاس‌ها'))}</h4>
      <ParamTable />
      <P className="small muted">
        {bi(
          'A gradient shows "?" until the backward pass reaches its layer. Bias gradients equal the neuron\'s $\\partial L/\\partial z$, because $\\partial z/\\partial b = 1$.',
          'هر گرادیان تا وقتی گذر رو به عقب به لایه‌اش نرسیده «?» نشان می‌دهد. گرادیان بایاس برابر $\\partial L/\\partial z$ همان نورون است، چون $\\partial z/\\partial b = 1$.',
        )}
      </P>
    </div>
  );
}

function CostTable() {
  const { tr } = useT();
  const [s, set] = useS();
  const net = useLab((st) => st.net);
  const here = sizes(net);
  const whatIf = [10, ...Array.from({ length: s.depth }, () => s.width), 1];
  const rows = [
    { name: tr(bi('this network', 'این شبکه')), sz: here },
    { name: tr(bi('what-if network', 'شبکه فرضی')), sz: whatIf },
  ].map((r) => ({ ...r, c: gradientCost(r.sz), paths: pathCount(r.sz, 1) }));
  return (
    <div className="stack" data-testid="l11-cost">
      <h4>{tr(bi('Cost of all gradients (multiply-adds, one example)', 'هزینه همه گرادیان‌ها (ضرب-جمع، یک نمونه)'))}</h4>
      <Slider label={tr(bi('hidden width', 'پهنای لایه پنهان'))} value={s.width} min={2} max={1024} step={1} onChange={(width) => set({ width: Math.round(width) })} digits={0} />
      <Slider label={tr(bi('hidden layers', 'تعداد لایه پنهان'))} value={s.depth} min={1} max={10} step={1} onChange={(depth) => set({ depth: Math.round(depth) })} digits={0} />
      <table className="tbl">
        <thead>
          <tr>
            <th />
            <th className="num">{tr(bi('parameters', 'پارامترها'))}</th>
            <th className="num">{tr(bi('finite diff. (2 passes / param)', 'تفاضل متناهی (۲ گذر برای هر پارامتر)'))}</th>
            <th className="num">{tr(bi('backprop (1 fwd + 1 bwd)', 'Backprop (۱ جلو + ۱ عقب)'))}</th>
            <th className="num">{tr(bi('ratio', 'نسبت'))}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name}>
              <td>
                {r.name}
                <div className="small muted mono">{r.sz.join('→')}</div>
              </td>
              <td className="num">{big(r.c.P)}</td>
              <td className="num">{big(r.c.fdCentral)}</td>
              <td className="num c-g">{big(r.c.backprop)}</td>
              <td className="num">{big(r.c.fdCentral / r.c.backprop)}×</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="stat-grid">
        {rows.map((r) => (
          <Stat key={r.name} k={tr(bi(`paths from one first-layer weight (${r.name})`, `مسیرها از یک وزن لایه اول (${r.name})`))} v={big(r.paths)} />
        ))}
      </div>
      <P className="small muted">
        {bi(
          'Finite differences: $2P$ forward passes of $F$ multiply-adds each. Backprop: $F$ forward plus about $2F$ backward, independent of $P$. The number of paths explodes with depth, but backprop never enumerates them.',
          'تفاضل متناهی: $2P$ گذر رو به جلو، هر کدام $F$ ضرب-جمع. Backprop: یک $F$ رو به جلو و حدود $2F$ رو به عقب، مستقل از $P$. تعداد مسیرها با عمق منفجر می‌شود، اما Backprop هیچ‌وقت آنها را تک‌تک نمی‌شمارد.',
        )}
      </P>
    </div>
  );
}

function Bottom() {
  const { tr } = useT();
  return (
    <div className="viz-grid">
      <div className="stack">
        <h4>{tr(bi('Numerical gradient check', 'بررسی عددی گرادیان'))}</h4>
        <P className="small muted">
          {bi(
            'For every parameter, nudge it by $\\pm h$ ($h = 10^{-5}$), re-run the forward pass, and compare $\\frac{L(\\theta+h)-L(\\theta-h)}{2h}$ with backprop. Rows marked "near a ReLU kink" can legitimately disagree.',
            'برای هر پارامتر، آن را به اندازه $\\pm h$ ($h = 10^{-5}$) تکان بده، گذر رو به جلو را دوباره اجرا کن و $\\frac{L(\\theta+h)-L(\\theta-h)}{2h}$ را با Backprop مقایسه کن. ردیف‌هایی که «نزدیک شکستگی ReLU» علامت خورده‌اند ممکن است به‌درستی با هم نخوانند.',
          )}
        </P>
        <GradCheckPanel />
      </div>
      <CostTable />
    </div>
  );
}

export const lesson11: LessonDef = {
  id: ID,
  title: bi('Backpropagation', 'پس‌انتشار (Backpropagation)'),
  short: bi('Backpropagation', 'Backprop'),
  objectives: [
    bi('Separate forward, loss, backward and update phases of one training step', 'مرحله‌های رو به جلو، خطا، رو به عقب و به‌روزرسانی یک گام آموزش را از هم جدا کنی'),
    bi('Compute ∂L/∂z layer by layer from right to left', '∂L/∂z را لایه به لایه از راست به چپ حساب کنی'),
    bi('Explain why contributions of converging paths are added', 'توضیح بدهی چرا سهم مسیرهایی که به هم می‌رسند جمع می‌شود'),
    bi('Verify backprop with a finite-difference gradient check', 'Backprop را با بررسی گرادیان به روش تفاضل متناهی تأیید کنی'),
    bi('Explain why backprop costs about one extra pass instead of one pass per parameter', 'توضیح بدهی چرا Backprop حدود یک گذر اضافه هزینه دارد، نه یک گذر به ازای هر پارامتر'),
  ],
  presetId: 'branching',
  revealSteps: 6,
  Explain,
  Center,
  Right,
  Bottom,
  checkpoint: {
    question: bi('A weight reaches the loss along two paths whose products are +3 and −1. What is ∂L/∂w?', 'یک وزن از دو مسیر به خطا می‌رسد که حاصل‌ضربشان ۳+ و ۱− است. ∂L/∂w چقدر است؟'),
    choices: [
      { id: 'a', label: bi('−3 (multiply the paths)', '۳− (مسیرها را ضرب کن)') },
      { id: 'b', label: bi('+2 (add the paths)', '۲+ (مسیرها را جمع کن)') },
      { id: 'c', label: bi('+3 (take the larger path)', '۳+ (مسیر بزرگ‌تر را بردار)') },
    ],
    answer: 'b',
    explain: bi(
      'Along one path, local derivatives multiply. Different paths are separate routes for the same small change, and their effects on the loss add up: 3 + (−1) = 2.',
      'در طول یک مسیر، مشتق‌های محلی در هم ضرب می‌شوند. مسیرهای مختلف راه‌های جداگانه یک تغییر کوچک‌اند و اثرشان روی خطا با هم جمع می‌شود: 3 + (−1) = 2.',
    ),
  },
};
