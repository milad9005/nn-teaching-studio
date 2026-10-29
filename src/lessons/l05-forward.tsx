// Lesson 05: Forward propagation on preset "branching" (1 -> 2 -> 2 -> 1, ReLU, ReLU, linear).
// Everything is read from the shared lab store: the canvas, the layer-by-layer stepper, the
// equations and the table all show the engine's trace for the same x and weights.

import { useEffect } from 'react';
import type { LessonDef } from './types';
import { bi, useT } from '../i18n';
import { useLab, useLabDerived, visibility } from '../state/lab';
import { useLessonState } from '../state/lessonState';
import { NetworkCanvas } from '../components/NetworkCanvas';
import { OutputPlot, ParamList } from '../components/LabPanels';
import { Tex } from '../components/Tex';
import { Segmented, Slider, Stat } from '../components/controls';
import { Advanced, Callout, Cycle, P, Predict, Reveal } from '../components/teaching';
import { fmt, tex, texP, type ActivationName, type Network, type Trace } from '../engine';

const ID = 'l05';

interface S {
  form: 'neuron' | 'matrix';
}
const INIT: S = { form: 'neuron' };
const useS = () => useLessonState<S>(ID, INIT);

const F: Record<ActivationName, string> = { linear: '\\operatorname{id}', relu: '\\relu', sigmoid: '\\sigma', tanh: '\\tanh', softmax: '\\operatorname{softmax}' };

const aNum = (l: number, v: number) => (l === 0 ? `\\cx{${texP(v)}}` : `\\ca{${texP(v)}}`);

/** z and a of neuron j in weight layer l (0-based), with the numbers substituted. */
function neuronTex(net: Network, tr: Trace, l: number, j: number): [string, string] {
  const layer = net.layers[l];
  const terms = layer.W[j].map((w, i) => `\\cw{${texP(w)}}\\cdot${aNum(l, tr.a[l][i])}`).join(' + ');
  const z = tr.z[l][j];
  return [
    `\\cz{z^{(${l + 1})}_{${j + 1}}} = ${terms} + \\cb{${texP(layer.b[j])}} = \\cz{${tex(z)}}`,
    `\\ca{a^{(${l + 1})}_{${j + 1}}} = ${F[layer.act]}(\\cz{${tex(z)}}) = \\ca{${tex(tr.a[l + 1][j])}}`,
  ];
}

const col = (v: number[], c: string) => `\\begin{bmatrix}${v.map((x) => `\\${c}{${tex(x)}}`).join('\\\\')}\\end{bmatrix}`;
function matrixTex(net: Network, tr: Trace, l: number): [string, string] {
  const layer = net.layers[l];
  const W = `\\begin{bmatrix}${layer.W.map((r) => r.map((w) => `\\cw{${tex(w)}}`).join(' & ')).join('\\\\')}\\end{bmatrix}`;
  return [
    `\\cz{\\mathbf{z}^{(${l + 1})}} = ${W}${col(tr.a[l], l === 0 ? 'cx' : 'ca')} + ${col(layer.b, 'cb')} = ${col(tr.z[l], 'cz')}`,
    `\\ca{\\mathbf{a}^{(${l + 1})}} = ${F[layer.act]}\\big(\\cz{\\mathbf{z}^{(${l + 1})}}\\big) = ${col(tr.a[l + 1], 'ca')}`,
  ];
}

/** The forward stepper as this lesson uses it: k = number of revealed activation layers, or null (all). */
function useForward() {
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const stepper = useLab((s) => s.stepper);
  const der = useLabDerived();
  const L = net.layers.length;
  const k = stepper && stepper.phase === 'forward' ? stepper.k : null;
  const vis = visibility(k === null ? null : stepper, L);
  return { net, data, L, k, vis, tr: der.one.trace, loss: der.one.loss, x: data.X[der.sample][0], y: data.Y[der.sample][0] };
}

const setK = (k: number) => useLab.setState({ stepper: { phase: 'forward', k } });

/**
 * This lesson is forward-only: "everything shown" is the last forward step (k = L + 2), not the
 * stepper's null state, so the canvas never shows gradient badges here. Restores null on leave.
 */
function useForwardOnly() {
  const stepper = useLab((s) => s.stepper);
  const L = useLab((s) => s.net.layers.length);
  useEffect(() => {
    if (!stepper || stepper.phase !== 'forward') setK(L + 2);
  }, [stepper, L]);
  useEffect(
    () => () => {
      if (useLab.getState().stepper?.phase === 'forward') useLab.getState().showAll();
    },
    [],
  );
}

function stepName(k: number | null, L: number) {
  if (k === null) return bi('All layers computed', 'همه لایه‌ها حساب شده‌اند');
  if (k === 1) return bi('Input: x is placed in layer 0', 'ورودی: x در لایه ۰ قرار می‌گیرد');
  if (k <= L) return bi(`Hidden layer ${k - 1} reads layer ${k - 2}`, `لایه پنهان ${k - 1} خروجی لایه ${k - 2} را می‌خواند`);
  if (k === L + 1) return bi(`Output layer reads layer ${L - 1}`, `لایه خروجی، لایه ${L - 1} را می‌خواند`);
  return bi('Loss: the output is compared with the target (everything computed)', 'خطا: خروجی با هدف مقایسه می‌شود (همه چیز حساب شده)');
}

function Explain() {
  const { tr } = useT();
  const k = useLab((s) => (s.stepper?.phase === 'forward' ? s.stepper.k : null));
  const L = useLab((s) => s.net.layers.length);
  const stepping = k !== null && k < L + 2;
  return (
    <>
      <Cycle active={stepping ? 2 : 0} />
      <Reveal at={1}>
        <h3>{tr(bi('1. Layers in a row', '۱. لایه‌ها پشت سر هم'))}</h3>
        <P>
          {bi(
            'This network has one input, two hidden layers of two ReLU neurons each, and one linear output neuron. Every layer is exactly what you built in lesson 4, with one twist: a layer\'s inputs are the outputs of the layer before it.',
            'این شبکه یک ورودی، دو لایه پنهان هر کدام با دو نورون ReLU و یک نورون خروجی خطی دارد. هر لایه دقیقاً همان چیزی است که در درس ۴ ساختی، با یک تفاوت: ورودی‌های هر لایه، خروجی‌های لایه قبلی هستند.',
          )}
        </P>
        <P>
          {bi(
            'Computing the output means walking from left to right, one layer at a time. That walk is forward propagation (the forward pass). In code it is a nested call: $\\hat y = f_3(f_2(f_1(x)))$.',
            'حساب کردن خروجی یعنی از چپ به راست، لایه به لایه جلو رفتن. به این مسیر انتشار رو به جلو (forward propagation یا forward pass) می‌گویند. در کد یک فراخوانی تودرتو است: $\\hat y = f_3(f_2(f_1(x)))$.',
          )}
        </P>
      </Reveal>
      <Reveal at={2}>
        <h3>{tr(bi('2. One neuron, one step', '۲. یک نورون، یک قدم'))}</h3>
        <P>
          {bi(
            'Inside layer $l$, neuron $j$ does the two things you already know: a weighted sum of the previous layer\'s outputs plus a bias (lesson 2), then an activation (lesson 3):',
            'در لایه $l$، نورون $j$ همان دو کاری را می‌کند که می‌شناسی: جمع وزن‌دار خروجی‌های لایه قبل به‌علاوه بایاس (درس ۲)، بعد یک تابع فعال‌سازی (درس ۳):',
          )}
        </P>
        <Tex display>{'\\cz{z^{(l)}_j} = \\sum_i \\cw{w^{(l)}_{ji}}\\,\\ca{a^{(l-1)}_i} + \\cb{b^{(l)}_j}, \\qquad \\ca{a^{(l)}_j} = f\\big(\\cz{z^{(l)}_j}\\big)'}</Tex>
        <P>
          {bi(
            'Here $a^{(0)} = x$. Press "Next layer" under the network to compute one layer at a time; the edges that carry numbers into the active layer pulse.',
            'اینجا $a^{(0)} = x$ است. دکمه «لایه بعد» زیر شبکه را بزن تا هر بار یک لایه حساب شود؛ یال‌هایی که عددها را به لایه فعال می‌برند چشمک می‌زنند.',
          )}
        </P>
        {!stepping && (
          <button className="btn sm primary" onClick={() => setK(1)} data-testid="start-forward">
            {tr(bi('Start the forward pass →', 'انتشار رو به جلو را شروع کن ←'))}
          </button>
        )}
      </Reveal>
      <Reveal at={3}>
        <Predict
          id={ID + '-double'}
          question={bi('At $x = 1$ the output is $\\hat y = 2.5$. If $x$ doubles to 2, what will $\\hat y$ be?', 'در $x = 1$ خروجی $\\hat y = 2.5$ است. اگر $x$ دو برابر شود و به ۲ برسد، $\\hat y$ چند می‌شود؟')}
          choices={[
            { id: '5', label: bi('5: it doubles too', '۵: آن هم دو برابر می‌شود') },
            { id: '4', label: bi('4: it grows, but does not double', '۴: بزرگ‌تر می‌شود ولی دو برابر نه') },
            { id: '2.5', label: bi('2.5: no change', '۲.۵: تغییری نمی‌کند') },
          ]}
          answer="4"
          explain={bi(
            'Set $x = 2$ and step through. Neuron $a^{(2)}_2$ stays at 0.5 because its two inputs cancel ($0.5\\cdot x - 1\\cdot 0.5x$), and the biases add constants, so for $x > 0$ this network computes $\\hat y = 1.5x + 1$: $2.5 \\to 4$.',
            '$x = 2$ را تنظیم کن و قدم به قدم جلو برو. نورون $a^{(2)}_2$ روی ۰.۵ می‌ماند چون دو ورودی‌اش همدیگر را خنثی می‌کنند ($0.5\\cdot x - 1\\cdot 0.5x$)، و بایاس‌ها مقدار ثابت اضافه می‌کنند؛ پس برای $x > 0$ این شبکه $\\hat y = 1.5x + 1$ را حساب می‌کند: $2.5 \\to 4$.',
          )}
        />
      </Reveal>
      <Reveal at={4}>
        <h3>{tr(bi('3. What flows between layers', '۳. چه چیزی بین لایه‌ها جریان دارد'))}</h3>
        <P>
          {bi(
            'Each layer turns its input list of numbers into a new list, a new description of the same example. The last layer turns that description into the prediction, and the loss compares it with the target. No weight changes during a forward pass: it only evaluates the function.',
            'هر لایه فهرست عددهای ورودی‌اش را به یک فهرست تازه تبدیل می‌کند؛ توصیف تازه‌ای از همان مثال. لایه آخر این توصیف را به پیش‌بینی تبدیل می‌کند و خطا آن را با هدف مقایسه می‌کند. در انتشار رو به جلو هیچ وزنی عوض نمی‌شود: فقط تابع حساب می‌شود.',
          )}
        </P>
        <Callout>
          {tr(
            bi(
              'Drag x below 0. Both neurons of layer 1 output 0, so everything after them sees zeros and only biases survive: the output stops depending on x (ŷ = 1 for every x < 0).',
              'x را به زیر ۰ ببر. هر دو نورون لایه ۱ صفر می‌دهند، پس همه لایه‌های بعدی فقط صفر می‌بینند و فقط بایاس‌ها باقی می‌مانند: خروجی دیگر به x بستگی ندارد (برای هر x < 0، ŷ = 1).',
            ),
          )}
        </Callout>
      </Reveal>
      <Reveal at={5}>
        <Advanced>
          <P>
            {bi(
              'Stack a layer\'s neurons and the sums become one matrix-vector product: $\\mathbf{z}^{(l)} = W^{(l)}\\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)}$, $\\mathbf{a}^{(l)} = f^{(l)}(\\mathbf{z}^{(l)})$ applied elementwise. Row $j$ of $W^{(l)}$ holds the weights into neuron $j$.',
              'اگر نورون‌های یک لایه را روی هم بچینیم، جمع‌ها یک ضرب ماتریس در بردار می‌شوند: $\\mathbf{z}^{(l)} = W^{(l)}\\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)}$ و $\\mathbf{a}^{(l)} = f^{(l)}(\\mathbf{z}^{(l)})$ که درایه به درایه اعمال می‌شود. سطر $j$ ماتریس $W^{(l)}$ وزن‌های ورودی به نورون $j$ را نگه می‌دارد.',
            )}
          </P>
          <P>
            {bi(
              'Shapes: $W^{(l)} \\in \\mathbb{R}^{n_l \\times n_{l-1}}$. Here $W^{(1)}$ is $2\\times1$, $W^{(2)}$ is $2\\times2$, $W^{(3)}$ is $1\\times2$, and $\\mathbf{b}^{(l)} \\in \\mathbb{R}^{n_l}$. For a batch with examples as columns, $Z^{(l)} = W^{(l)}A^{(l-1)} + \\mathbf{b}^{(l)}\\mathbf{1}^\\top$. A forward pass costs $\\sum_l n_l\\,n_{l-1}$ multiply-adds per example.',
              'ابعاد: $W^{(l)} \\in \\mathbb{R}^{n_l \\times n_{l-1}}$. اینجا $W^{(1)}$ از نوع $2\\times1$، $W^{(2)}$ از نوع $2\\times2$ و $W^{(3)}$ از نوع $1\\times2$ است و $\\mathbf{b}^{(l)} \\in \\mathbb{R}^{n_l}$. برای یک دسته (batch) که مثال‌ها ستون‌هایش هستند، $Z^{(l)} = W^{(l)}A^{(l-1)} + \\mathbf{b}^{(l)}\\mathbf{1}^\\top$. هزینه یک forward pass برای هر مثال $\\sum_l n_l\\,n_{l-1}$ ضرب و جمع است.',
            )}
          </P>
        </Advanced>
      </Reveal>
    </>
  );
}

function ForwardControls() {
  const { tr } = useT();
  const { k, L } = useForward();
  const last = L + 2;
  const cur = k ?? last;
  return (
    <div className="run-controls" data-testid="forward-controls">
      <div className="row">
        <button className="btn" onClick={() => setK(1)} data-testid="fw-restart">
          ↺ {tr(bi('Start over', 'از اول'))}
        </button>
        <button className="btn" onClick={() => setK(Math.max(1, cur - 1))} disabled={cur <= 1} data-testid="fw-back">
          {tr(bi('← Back', 'قبلی →'))}
        </button>
        <button className="btn primary" onClick={() => setK(cur + 1)} disabled={cur >= last} data-testid="fw-next">
          {tr(bi('Next layer →', 'لایه بعد ←'))}
        </button>
        <button className="btn ghost sm" onClick={() => setK(last)} disabled={cur >= last}>
          {tr(bi('Show all', 'نمایش همه'))}
        </button>
      </div>
      <div className="small stepper-label" data-testid="fw-label">
        {tr(bi('Step', 'قدم'))} {cur}/{last} · {tr(stepName(cur, L))}
      </div>
    </div>
  );
}

/** What the current step computes, shown right under the network. */
function ActiveStep() {
  const { tr } = useT();
  const { net, k, L, tr: trace, x, y, loss } = useForward();
  if (k === null) return null;
  if (k === 1) return <Tex display>{`\\ca{a^{(0)}} = \\cx{x} = \\cx{${tex(x)}}`}</Tex>;
  if (k > L + 1)
    return <Tex display>{`\\cL{L} = (\\cyhat{\\hat y} - \\cy{y})^2 = (${tex(trace.a[L][0])} - ${texP(y)})^2 = \\cL{${tex(loss)}}`}</Tex>;
  const l = k - 2;
  return (
    <div className="card flat" data-testid="active-step">
      <div className="small muted">
        {tr(bi(`Inputs of this layer = outputs of layer ${l}:`, `ورودی‌های این لایه = خروجی‌های لایه ${l}:`))}{' '}
        <Tex>{`${l === 0 ? '\\cx{x}' : `\\ca{\\mathbf{a}^{(${l})}}`} = (${trace.a[l].map((v) => tex(v)).join(',\\ ')})`}</Tex>
      </div>
      {net.layers[l].b.map((_, j) =>
        neuronTex(net, trace, l, j).map((t, q) => <Tex key={`${j}-${q}`} display>{t}</Tex>),
      )}
    </div>
  );
}

function Center() {
  useForwardOnly();
  return (
    <div className="center-stack">
      <div className="canvas-wrap" style={{ minHeight: 260 }}>
        <NetworkCanvas />
      </div>
      <ForwardControls />
      <ActiveStep />
    </div>
  );
}

function Right() {
  const { tr } = useT();
  const [s, set] = useS();
  const { net, L, vis, tr: trace, x, y, loss } = useForward();
  const setInput = useLab((st) => st.setInput);
  const setTarget = useLab((st) => st.setTarget);
  return (
    <div className="stack">
      <Slider label="x" value={x} min={-3} max={3} step={0.1} color="x" onChange={(v) => setInput(0, 0, v)} testId="l05-x" />
      <Slider label={tr(bi('target y', 'هدف y'))} value={y} min={-3} max={6} step={0.1} color="y" onChange={(v) => setTarget(0, 0, v)} />
      <Segmented
        label="form"
        value={s.form}
        onChange={(form) => set({ form })}
        options={[
          { value: 'neuron', label: tr(bi('Neuron by neuron', 'نورون به نورون')) },
          { value: 'matrix', label: tr(bi('Matrix form', 'شکل ماتریسی')) },
        ]}
      />
      {net.layers.map((layer, l) => (
        <div key={l} className="stack" style={{ gap: 2 }} data-testid={`layer-eq-${l + 1}`}>
          <h4>
            {l === L - 1 ? tr(bi(`Layer ${l + 1} (output, ${layer.act})`, `لایه ${l + 1} (خروجی، ${layer.act})`)) : tr(bi(`Layer ${l + 1} (hidden, ${layer.act})`, `لایه ${l + 1} (پنهان، ${layer.act})`))}
          </h4>
          {vis.act(l + 1) ? (
            s.form === 'neuron' ? (
              layer.b.map((_, j) => neuronTex(net, trace, l, j).map((t, q) => <Tex key={`${j}-${q}`} display>{t}</Tex>))
            ) : (
              matrixTex(net, trace, l).map((t, q) => <Tex key={q} display>{t}</Tex>)
            )
          ) : (
            <p className="small muted">{tr(bi('Not computed yet: step forward to reach this layer.', 'هنوز حساب نشده: جلو برو تا به این لایه برسی.'))}</p>
          )}
        </div>
      ))}
      <h4>{tr(bi('Loss', 'خطا'))}</h4>
      {vis.loss ? (
        <Tex display>{`\\cL{L} = (\\cyhat{${tex(trace.a[L][0])}} - \\cy{${texP(y)}})^2 = \\cL{${tex(loss)}}`}</Tex>
      ) : (
        <p className="small muted">{tr(bi('Computed after the output.', 'بعد از خروجی حساب می‌شود.'))}</p>
      )}
      <div className="stat-grid">
        <Stat k="x" v={fmt(x)} color="x" />
        <Stat k="ŷ" v={vis.act(L) ? fmt(trace.a[L][0], 4) : '?'} color="yhat" testId="l05-yhat" />
        <Stat k="L" v={vis.loss ? fmt(loss, 4) : '?'} color="L" />
      </div>
      <details>
        <summary className="small">{tr(bi('Edit every weight and bias', 'ویرایش همه وزن‌ها و بایاس‌ها'))}</summary>
        <P className="small muted">
          {bi(
            'The list counts neurons from 0: W2[0,1] is the weight from neuron 2 of layer 1 into neuron 1 of layer 2, written $w^{(2)}_{12}$ in the equations.',
            'این فهرست شماره نورون‌ها را از ۰ می‌شمارد: W2[0,1] وزنی است که از نورون ۲ لایه ۱ به نورون ۱ لایه ۲ می‌رود و در معادله‌ها به شکل $w^{(2)}_{12}$ نوشته می‌شود.',
          )}
        </P>
        <ParamList showGrads={false} />
      </details>
    </div>
  );
}

function Bottom() {
  const { tr } = useT();
  const { net, data, L, k, vis, tr: trace, loss } = useForward();
  const sizes = [net.inputSize, ...net.layers.map((ly) => ly.b.length)];
  const vec = (v: number[]) => `(${v.map((q) => fmt(q, 3)).join(', ')})`;
  const active = k === null ? -1 : k - 1;
  return (
    <div className="viz-grid">
      <table className="tbl" data-testid="layer-table">
        <thead>
          <tr>
            <th>{tr(bi('layer', 'لایه'))}</th>
            <th>{tr(bi('weights', 'وزن‌ها'))}</th>
            <th className="num c-z">z</th>
            <th>f</th>
            <th className="num c-a">a</th>
          </tr>
        </thead>
        <tbody>
          <tr className={active === 0 ? 'hl' : ''}>
            <td>0 · {tr(bi('input', 'ورودی'))}</td>
            <td className="muted">—</td>
            <td className="num muted">—</td>
            <td className="muted">—</td>
            <td className="num c-x" dir="ltr">
              {vec(trace.a[0])}
            </td>
          </tr>
          {net.layers.map((layer, l) => (
            <tr key={l} className={active === l + 1 ? 'hl' : ''}>
              <td>
                {l + 1} · {l === L - 1 ? tr(bi('output', 'خروجی')) : tr(bi('hidden', 'پنهان'))}
              </td>
              <td className="mono small" dir="ltr">
                W{l + 1}: {sizes[l + 1]}×{sizes[l]}
              </td>
              <td className="num" dir="ltr">
                {vis.act(l + 1) ? vec(trace.z[l]) : '?'}
              </td>
              <td>{layer.act}</td>
              <td className="num" dir="ltr">
                {vis.act(l + 1) ? vec(trace.a[l + 1]) : '?'}
              </td>
            </tr>
          ))}
          <tr className={active === L + 1 ? 'hl' : ''}>
            <td>{tr(bi('loss', 'خطا'))}</td>
            <td className="small muted" dir="ltr">
              y = {fmt(data.Y[0][0])}
            </td>
            <td />
            <td>MSE</td>
            <td className="num c-L">{vis.loss ? fmt(loss, 4) : '?'}</td>
          </tr>
        </tbody>
      </table>
      <div>
        <div className="small muted">{tr(bi('The whole function this network computes: ŷ for every x (others fixed)', 'کل تابعی که این شبکه حساب می‌کند: ŷ برای هر x (بقیه ثابت)'))}</div>
        <OutputPlot height={200} />
      </div>
    </div>
  );
}

export const lesson05: LessonDef = {
  id: ID,
  title: bi('Forward Propagation', 'انتشار رو به جلو'),
  short: bi('Forward pass', 'انتشار رو به جلو'),
  objectives: [
    bi('Compute a network\'s output one layer at a time', 'خروجی یک شبکه را لایه به لایه حساب کنی'),
    bi('Explain how the outputs of one layer become the inputs of the next', 'توضیح بدهی خروجی‌های یک لایه چطور ورودی‌های لایه بعد می‌شوند'),
    bi('Write each neuron as z = Σ w·a + b and a = f(z), with real numbers', 'هر نورون را به شکل z = Σ w·a + b و a = f(z) با عددهای واقعی بنویسی'),
    bi('Read the same computation in matrix form and check the shapes', 'همین محاسبه را به شکل ماتریسی بخوانی و ابعاد را بررسی کنی'),
  ],
  presetId: 'branching',
  revealSteps: 5,
  Explain,
  Center,
  Right,
  Bottom,
  checkpoint: {
    question: bi('During a forward pass, what does hidden layer 2 receive as its inputs?', 'در انتشار رو به جلو، لایه پنهان ۲ چه چیزی را به عنوان ورودی دریافت می‌کند؟'),
    choices: [
      { id: 'a', label: bi('The original input x, directly', 'مستقیم خود ورودی x را') },
      { id: 'b', label: bi('The activations $a^{(1)}$ computed by layer 1', 'خروجی‌های فعال‌شده $a^{(1)}$ که لایه ۱ حساب کرده') },
      { id: 'c', label: bi('The loss', 'خطا را') },
    ],
    answer: 'b',
    explain: bi(
      'Each layer only sees the previous layer\'s outputs: $\\mathbf{z}^{(2)} = W^{(2)}\\mathbf{a}^{(1)} + \\mathbf{b}^{(2)}$. The input $x$ reaches layer 2 only through layer 1.',
      'هر لایه فقط خروجی‌های لایه قبلی را می‌بیند: $\\mathbf{z}^{(2)} = W^{(2)}\\mathbf{a}^{(1)} + \\mathbf{b}^{(2)}$. ورودی $x$ فقط از راه لایه ۱ به لایه ۲ می‌رسد.',
    ),
  },
};
