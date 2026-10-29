// Lesson 18: Training vs inference. Uses the shared lab store (preset "temperature": predict
// cooling power from outside temperature with one linear neuron). Training mode runs the full
// loop through the lab (forward → loss → backward → update); inference mode only reads the
// learned weights and runs forward passes. The closing overview of CNNs and Transformers is a
// static conceptual diagram, clearly labelled as not computed.

import { useMemo } from 'react';
import type { LessonDef } from './types';
import { bi, useT, type Bi, type StringKey } from '../i18n';
import { useLab, useLabDerived } from '../state/lab';
import { useLessonState } from '../state/lessonState';
import { NetworkCanvas } from '../components/NetworkCanvas';
import { Plot2D, type PlotLayer } from '../components/Plot2D';
import { Tex } from '../components/Tex';
import { Segmented, Slider, Stat } from '../components/controls';
import { Advanced, Callout, Cycle, P, Predict, Reveal } from '../components/teaching';
import { LossHistory, RunControls, TrainStats } from '../components/LabPanels';
import { fmt, predict, sizes, tex, texP } from '../engine';
import { iterationCost } from './math/l18';

const ID = 'l18';
type Mode = 'train' | 'infer' | 'bigger';
interface S {
  mode: Mode;
  x: number;
}
const INIT: S = { mode: 'train', x: 3.2 };
const useS = () => useLessonState<S>(ID, INIT);
const QUERIES = [2.2, 2.8, 3.2, 3.8, 4.2];

type Phase = 'forward' | 'loss' | 'backward' | 'update' | null;
const PHASES: { id: Exclude<Phase, null>; label: Bi }[] = [
  { id: 'forward', label: bi('Forward', 'رو به جلو (forward)') },
  { id: 'loss', label: bi('Loss', 'خطا (loss)') },
  { id: 'backward', label: bi('Backward', 'رو به عقب (backward)') },
  { id: 'update', label: bi('Update', 'به‌روزرسانی (update)') },
];

function usePhase(): Phase {
  const stepper = useLab((s) => s.stepper);
  const L = useLab((s) => s.net.layers.length);
  if (!stepper) return null;
  if (stepper.phase === 'forward') return stepper.k > L + 1 ? 'loss' : 'forward';
  return stepper.phase;
}

function Explain() {
  const { tr } = useT();
  return (
    <>
      <Cycle active={3} />
      <Reveal at={1}>
        <h3>{tr(bi('1. Two phases in a model\'s life', '۱. دو مرحله در زندگی یک مدل'))}</h3>
        <P>
          {bi(
            'Training: the model looks at examples with known answers and adjusts its weights. Inference: training is over, the weights are frozen, and the model is used to answer new questions. Training happens once (and is expensive); inference happens every time someone uses the model.',
            'آموزش (training): مدل مثال‌هایی با جواب معلوم را می‌بیند و وزن‌هایش را تنظیم می‌کند. استنتاج (inference): آموزش تمام شده، وزن‌ها ثابت (frozen) شده‌اند و از مدل برای جواب دادن به سؤال‌های جدید استفاده می‌شود. آموزش یک بار انجام می‌شود (و گران است)؛ استنتاج هر بار که کسی از مدل استفاده می‌کند.',
          )}
        </P>
      </Reveal>
      <Reveal at={2}>
        <h3>{tr(bi('2. The training loop', '۲. حلقه آموزش'))}</h3>
        <P>
          {bi(
            'Everything from the previous lessons fits in one loop, repeated for every batch of every epoch: a forward pass computes $\\cyhat{\\hat y}$, the loss compares it with $\\cy{y}$, backpropagation computes $\\cg{\\partial L/\\partial \\theta}$, and the optimizer updates $\\theta$. Use "Forward step", "Backward step" and "Optimizer step" to walk through one turn, then "Train" to repeat it.',
            'همه چیز از درس‌های قبل در یک حلقه جا می‌شود که برای هر دسته از هر epoch تکرار می‌شود: forward pass مقدار $\\cyhat{\\hat y}$ را حساب می‌کند، خطا آن را با $\\cy{y}$ مقایسه می‌کند، backpropagation مقدار $\\cg{\\partial L/\\partial \\theta}$ را حساب می‌کند و بهینه‌ساز $\\theta$ را به‌روز می‌کند. با «گام رو به جلو»، «گام رو به عقب» و «گام بهینه‌ساز» یک دور را قدم به قدم برو، بعد با «آموزش» تکرارش کن.',
          )}
        </P>
      </Reveal>
      <Reveal at={3}>
        <Predict
          id={ID + '-infer'}
          question={bi('A trained model receives tomorrow\'s forecast. Which steps run?', 'یک مدل آموزش‌دیده پیش‌بینی هوای فردا را دریافت می‌کند. کدام مراحل اجرا می‌شوند؟')}
          choices={[
            { id: 'a', label: bi('Forward, loss, backward and update', 'forward، خطا، backward و به‌روزرسانی') },
            { id: 'b', label: bi('Forward only', 'فقط forward') },
            { id: 'c', label: bi('Forward and loss', 'forward و خطا') },
          ]}
          answer="b"
          explain={bi(
            'There is no target $y$ for tomorrow (that is what we want to know), so there is no loss, so nothing to differentiate and nothing to update. Switch to "Inference" and move the input slider: one forward pass per input.',
            'برای فردا هدف $y$ وجود ندارد (این همان چیزی است که می‌خواهیم بدانیم)، پس خطایی نیست، پس چیزی برای مشتق گرفتن و به‌روز کردن نیست. به «استنتاج» برو و اسلایدر ورودی را حرکت بده: برای هر ورودی یک forward pass.',
          )}
        />
      </Reveal>
      <Reveal at={4}>
        <h3>{tr(bi('3. Inference', '۳. استنتاج'))}</h3>
        <P>
          {bi(
            'In inference the weights are read-only. No gradients are computed and nothing needs to be remembered for a backward pass, so one prediction costs exactly one forward pass. In PyTorch this is model.eval() with torch.no_grad().',
            'در استنتاج وزن‌ها فقط‌خواندنی‌اند. هیچ گرادیانی حساب نمی‌شود و لازم نیست چیزی برای backward نگه داشته شود، پس هر پیش‌بینی دقیقاً به اندازه یک forward pass هزینه دارد. در PyTorch این همان model.eval() همراه با torch.no_grad() است.',
          )}
        </P>
      </Reveal>
      <Reveal at={5}>
        <h3>{tr(bi('4. The same ideas, bigger models', '۴. همین ایده‌ها، مدل‌های بزرگ‌تر'))}</h3>
        <P>
          {bi(
            'Open "Bigger models". A convolutional network (CNN) uses layers whose weights are small filters reused at every position of an image. A Transformer stacks attention layers (each token mixes in information from other tokens) and MLP blocks like the ones you built. Both have a forward pass, a loss, backpropagation and an optimizer such as Adam, trained with exactly this loop, and both run inference as forward passes.',
            '«مدل‌های بزرگ‌تر» را باز کن. شبکه کانولوشنی (CNN) لایه‌هایی دارد که وزن‌هایشان فیلترهای کوچکی‌اند که در همه جای تصویر دوباره استفاده می‌شوند. Transformer لایه‌های attention (هر توکن اطلاعاتی از توکن‌های دیگر را با خودش ترکیب می‌کند) و بلوک‌های MLP مثل همان‌هایی که ساختی را روی هم می‌چیند. هر دو forward pass، خطا، backpropagation و بهینه‌سازی مثل Adam دارند، با همین حلقه آموزش می‌بینند و استنتاجشان هم forward pass است.',
          )}
        </P>
        <Callout>
          {tr(
            bi(
              'You now know the core of how every modern neural network learns. Bigger models change the layers and the scale, not the loop.',
              'حالا هسته اصلی یادگیری همه شبکه‌های عصبی مدرن را می‌دانی. مدل‌های بزرگ‌تر لایه‌ها و مقیاس را عوض می‌کنند، نه حلقه را.',
            ),
          )}
        </Callout>
      </Reveal>
      <Reveal at={6}>
        <Advanced title={bi('Cost, memory and train-only behaviour', 'هزینه، حافظه و رفتارهای مخصوص آموزش')}>
          <P>
            {bi(
              'For a dense layer with $n_{in}$ inputs and $n_{out}$ outputs, the forward pass costs $n_{in}n_{out}$ multiply-adds per example. Backward needs the same again for $\\partial L/\\partial W$ and again for $\\partial L/\\partial a$ (passed to the layer below), so a training step costs roughly $3\\times$ a forward pass per example, plus one update per parameter. Training must also keep every layer\'s activations until the backward pass uses them; inference can discard them immediately.',
              'برای یک لایه dense با $n_{in}$ ورودی و $n_{out}$ خروجی، forward برای هر نمونه $n_{in}n_{out}$ ضرب‌وجمع هزینه دارد. backward یک بار دیگر همین مقدار برای $\\partial L/\\partial W$ و یک بار دیگر برای $\\partial L/\\partial a$ (که به لایه پایینی فرستاده می‌شود) لازم دارد، پس یک قدم آموزش برای هر نمونه تقریباً $3\\times$ یک forward است، به‌علاوه یک به‌روزرسانی برای هر پارامتر. آموزش باید فعال‌سازی‌های هر لایه را تا وقتی backward از آنها استفاده کند نگه دارد؛ استنتاج می‌تواند فوراً دورشان بریزد.',
            )}
          </P>
          <P>
            {bi(
              'Some layers behave differently in the two modes: dropout is active only in training, and batch normalization uses batch statistics in training but stored running averages in inference. A language model generates text by running inference repeatedly, one forward pass per new token.',
              'بعضی لایه‌ها در دو حالت رفتار متفاوتی دارند: dropout فقط در آموزش فعال است، و batch normalization در آموزش از آمار همان دسته و در استنتاج از میانگین‌های ذخیره‌شده استفاده می‌کند. یک مدل زبانی با اجرای مکرر استنتاج متن تولید می‌کند: برای هر توکن جدید یک forward pass.',
            )}
          </P>
        </Advanced>
      </Reveal>
    </>
  );
}

function LoopDiagram({ phase, active }: { phase: Phase; active: boolean }) {
  const { tr } = useT();
  return (
    <div className="factor-row" data-testid="loop-diagram">
      {PHASES.map((p, i) => (
        <div key={p.id} style={{ display: 'contents' }}>
          {i > 0 && <span className="factor-op">→</span>}
          <div className={`factor ${phase === p.id || (active && !phase) ? 'on' : ''}`} style={{ cursor: 'default', minWidth: 104 }}>
            <div style={{ fontWeight: 600 }}>{tr(p.label)}</div>
          </div>
        </div>
      ))}
      <span className="factor-op" title={tr(bi('repeat', 'تکرار'))}>↺</span>
    </div>
  );
}

function InferenceView() {
  const { t, tr } = useT();
  const [s, set] = useS();
  const net = useLab((st) => st.net);
  const data = useLab((st) => st.data);
  const units = useLab((st) => st.units);
  const w = net.layers[0].W[0][0];
  const b = net.layers[0].b[0];
  const yhat = predict(net, [s.x])[0];
  const layers = useMemo<PlotLayer[]>(
    () => [
      { kind: 'fn', f: (v) => predict(net, [v])[0], color: 'yhat', width: 2.4 },
      { kind: 'points', color: 'y', pts: data.X.map((x, n) => ({ x: x[0], y: data.Y[n][0], r: 3.5, hollow: true })) },
      { kind: 'vline', at: s.x, color: 'x' },
      { kind: 'points', color: 'yhat', pts: [{ x: s.x, y: yhat, r: 7, label: `ŷ = ${fmt(yhat, 3)}` }] },
    ],
    [net, data, s.x, yhat],
  );
  return (
    <>
      <Callout>
        {tr(bi(`Weights frozen: w = ${fmt(w, 4)}, b = ${fmt(b, 4)}. Nothing on this tab writes to them.`, `وزن‌ها ثابت‌اند: w = ${fmt(w, 4)}، b = ${fmt(b, 4)}. هیچ چیز در این زبانه آنها را تغییر نمی‌دهد.`))}
      </Callout>
      <Slider label={tr(bi('new input x', 'ورودی جدید x'))} value={s.x} min={1.5} max={4.5} step={0.05} color="x" onChange={(x) => set({ x })} testId="inf-x" />
      <Tex className="formula-hero" display>{`\\cyhat{\\hat y} = \\cw{${tex(w, 3)}}\\cdot\\cx{${tex(s.x, 2)}} + \\cb{${texP(b, 3)}} = \\cyhat{${tex(yhat, 4)}}`}</Tex>
      <Plot2D testId="inf-plot" xDomain={[1.5, 4.5]} yDomain="auto" layers={layers} height={250} xLabel={units ? t(units.x as StringKey) : 'x'} yLabel={units ? t(units.y as StringKey) : 'ŷ'} />
      <table className="tbl" data-testid="inf-batch">
        <thead>
          <tr>
            <th>{tr(bi('new input x', 'ورودی جدید x'))}</th>
            {QUERIES.map((q) => (
              <th key={q} className="num c-x">{fmt(q, 2)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>{tr(bi('prediction ŷ', 'پیش‌بینی ŷ'))}</td>
            {QUERIES.map((q) => (
              <td key={q} className="num c-yhat">{fmt(predict(net, [q])[0], 3)}</td>
            ))}
          </tr>
        </tbody>
      </table>
      <P className="small muted">{bi('Five predictions = five forward passes. No target, no loss, no gradient, no update.', 'پنج پیش‌بینی = پنج forward pass. بدون هدف، بدون خطا، بدون گرادیان، بدون به‌روزرسانی.')}</P>
    </>
  );
}

function BiggerView() {
  const { tr } = useT();
  const ink = 'var(--ink-2)';
  const box = { fill: 'var(--surface-2)', stroke: 'var(--line-strong)' };
  return (
    <>
      <div className="small muted">{tr(bi('Conceptual diagram only: nothing here is computed.', 'فقط یک نمودار مفهومی: هیچ چیز اینجا محاسبه نمی‌شود.'))}</div>
      <div className="viz-grid" data-testid="bigger">
        <div className="card flat">
          <h4>{tr(bi('MLP (what you built)', 'MLP (همانی که ساختی)'))}</h4>
          <svg viewBox="0 0 200 120" width="100%" role="img" aria-label="MLP">
            {[0, 1, 2].map((c) =>
              [0, 1, 2, 3].slice(0, c === 1 ? 4 : c === 0 ? 2 : 1).map((r, _, arr) => {
                const y = 60 + (r - (arr.length - 1) / 2) * 26;
                return <circle key={`${c}${r}`} cx={30 + c * 70} cy={y} r={9} fill={c === 0 ? 'var(--c-x)' : c === 1 ? 'var(--c-a)' : 'var(--c-yhat)'} />;
              }),
            )}
            {[0, 1].map((r) => [0, 1, 2, 3].map((q) => <line key={`a${r}${q}`} x1={39} y1={60 + (r - 0.5) * 26} x2={91} y2={60 + (q - 1.5) * 26} stroke="var(--c-w)" strokeWidth={1} opacity={0.6} />))}
            {[0, 1, 2, 3].map((q) => <line key={`b${q}`} x1={109} y1={60 + (q - 1.5) * 26} x2={161} y2={60} stroke="var(--c-w)" strokeWidth={1} opacity={0.6} />)}
          </svg>
          <P className="small">{bi('Every input connects to every unit with its own weight.', 'هر ورودی با وزن مخصوص خودش به همه نورون‌ها وصل است.')}</P>
        </div>
        <div className="card flat">
          <h4>{tr(bi('CNN', 'CNN (شبکه کانولوشنی)'))}</h4>
          <svg viewBox="0 0 200 120" width="100%" role="img" aria-label="CNN">
            {Array.from({ length: 36 }, (_, i) => (
              <rect key={i} x={10 + (i % 6) * 14} y={18 + Math.floor(i / 6) * 14} width={13} height={13} {...box} />
            ))}
            <rect x={10 + 14} y={18 + 14} width={41} height={41} fill="var(--c-w)" opacity={0.25} stroke="var(--c-w)" strokeWidth={2} />
            <line x1={66} y1={52} x2={122} y2={52} stroke={ink} markerEnd="url(#l18arr)" />
            <defs>
              <marker id="l18arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M0,0 L10,5 L0,10 z" fill={ink} />
              </marker>
            </defs>
            {Array.from({ length: 16 }, (_, i) => (
              <rect key={i} x={130 + (i % 4) * 14} y={32 + Math.floor(i / 4) * 14} width={13} height={13} {...box} fill={i === 5 ? 'var(--c-a)' : box.fill} />
            ))}
          </svg>
          <P className="small">{bi('One small 3×3 filter of weights slides over the whole image: the same weights are shared at every position.', 'یک فیلتر کوچک ۳×۳ از وزن‌ها روی کل تصویر می‌لغزد: همان وزن‌ها در همه جا مشترک‌اند.')}</P>
        </div>
        <div className="card flat">
          <h4>{tr(bi('Transformer', 'Transformer'))}</h4>
          <svg viewBox="0 0 200 120" width="100%" role="img" aria-label="Transformer">
            {[0, 1, 2, 3].map((i) => (
              <rect key={i} x={22 + i * 42} y={98} width={30} height={16} rx={3} fill="var(--c-x)" opacity={0.8} />
            ))}
            <rect x={14} y={60} width={172} height={26} rx={5} {...box} />
            <text x={100} y={77} textAnchor="middle" fontSize={11} fill={ink}>attention</text>
            <rect x={14} y={24} width={172} height={26} rx={5} {...box} />
            <text x={100} y={41} textAnchor="middle" fontSize={11} fill={ink}>MLP</text>
            {[0, 1, 2, 3].map((i) => [0, 1, 2, 3].map((j) => <line key={`${i}${j}`} x1={37 + i * 42} y1={98} x2={37 + j * 42} y2={86} stroke="var(--c-w)" opacity={0.35} />))}
            <text x={192} y={16} textAnchor="end" fontSize={11} fill={ink}>× N</text>
          </svg>
          <P className="small">{bi('Attention lets each token gather information from the others; an MLP block then processes each token. The pair is stacked N times.', 'Attention به هر توکن اجازه می‌دهد از بقیه اطلاعات جمع کند؛ بعد یک بلوک MLP هر توکن را پردازش می‌کند. این جفت N بار روی هم چیده می‌شود.')}</P>
        </div>
      </div>
      <table className="tbl">
        <thead>
          <tr>
            <th />
            <th>MLP</th>
            <th>CNN</th>
            <th>Transformer</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>{tr(bi('layers', 'لایه‌ها'))}</td>
            <td>{tr(bi('dense', 'dense'))}</td>
            <td>{tr(bi('convolution + dense', 'کانولوشن + dense'))}</td>
            <td>{tr(bi('attention + MLP', 'attention + MLP'))}</td>
          </tr>
          {[bi('forward pass', 'forward pass'), bi('loss', 'خطا'), bi('backpropagation', 'backpropagation'), bi('gradient-based optimizer', 'بهینه‌ساز مبتنی بر گرادیان')].map((r, i) => (
            <tr key={i}>
              <td>{tr(r)}</td>
              <td>✓</td>
              <td>✓</td>
              <td>✓</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function Center() {
  const { tr } = useT();
  const [s, set] = useS();
  const running = useLab((st) => st.running);
  const setRunning = useLab((st) => st.setRunning);
  const phase = usePhase();
  return (
    <div className="viz">
      <Segmented<Mode>
        label="mode"
        value={s.mode}
        onChange={(mode) => {
          if (mode !== 'train') setRunning(false);
          set({ mode });
        }}
        options={[
          { value: 'train', label: tr(bi('Training', 'آموزش')) },
          { value: 'infer', label: tr(bi('Inference', 'استنتاج')) },
          { value: 'bigger', label: tr(bi('Bigger models', 'مدل‌های بزرگ‌تر')) },
        ]}
      />
      {s.mode === 'train' && (
        <div className="center-stack">
          <LoopDiagram phase={phase} active={running} />
          <div className="canvas-wrap" style={{ minHeight: 220 }}>
            <NetworkCanvas />
          </div>
          <RunControls />
        </div>
      )}
      {s.mode === 'infer' && <InferenceView />}
      {s.mode === 'bigger' && <BiggerView />}
    </div>
  );
}

function Right() {
  const { tr } = useT();
  const [s] = useS();
  const net = useLab((st) => st.net);
  const data = useLab((st) => st.data);
  const opt = useLab((st) => st.opt);
  const der = useLabDerived();
  const sz = sizes(net);
  const N = data.X.length;
  const B = der.nextIdx.length;
  const here = iterationCost(sz, B);
  const digits = iterationCost([784, 128, 10], 32);
  const yes = <span className="c-pos">✓</span>;
  const no = <span className="c-neg">✗</span>;
  const rows: [Bi, boolean, boolean][] = [
    [bi('forward pass', 'forward pass'), true, true],
    [bi('needs the target y', 'به هدف y نیاز دارد'), true, false],
    [bi('loss', 'خطا'), true, false],
    [bi('backward pass (gradients)', 'backward (گرادیان‌ها)'), true, false],
    [bi('parameter update', 'به‌روزرسانی پارامترها'), true, false],
    [bi('weights change', 'وزن‌ها تغییر می‌کنند'), true, false],
    [bi('keeps activations for backward', 'فعال‌سازی‌ها را برای backward نگه می‌دارد'), true, false],
  ];
  const w = net.layers[0].W[0][0];
  const b = net.layers[0].b[0];
  const gw = der.full.grads.W[0][0][0];
  const gb = der.full.grads.b[0][0];
  return (
    <div className="stack">
      <table className="tbl" data-testid="mode-table">
        <thead>
          <tr>
            <th />
            <th className={s.mode === 'train' ? 'c-g' : ''}>{tr(bi('training', 'آموزش'))}</th>
            <th className={s.mode === 'infer' ? 'c-yhat' : ''}>{tr(bi('inference', 'استنتاج'))}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, a, i], n) => (
            <tr key={n}>
              <td>{tr(k)}</td>
              <td>{a ? yes : no}</td>
              <td>{i ? yes : no}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h4>{tr(bi('Cost in multiply-adds', 'هزینه بر حسب ضرب‌وجمع'))}</h4>
      <div className="stat-grid">
        <Stat k={tr(bi(`1 training iteration (B = ${B})`, `۱ تکرار آموزش (B = ${B})`))} v={here.training} color="g" testId="cost-train" />
        <Stat k={tr(bi('1 prediction', '۱ پیش‌بینی'))} v={here.inference} color="yhat" testId="cost-infer" />
      </div>
      <P className="small muted">
        {bi(
          `This network (${sz.join('→')}): ${B} × (forward ${here.cost.forward} + backward ${here.cost.backward}) + update ${here.cost.update}. The same count for a 784→128→10 digit classifier with batch 32: ${digits.training.toLocaleString('en')} per iteration vs ${digits.inference.toLocaleString('en')} per prediction.`,
          `این شبکه (${sz.join('→')}): ${B} × (forward ${here.cost.forward} + backward ${here.cost.backward}) + به‌روزرسانی ${here.cost.update}. همین شمارش برای یک دسته‌بند ارقام 784→128→10 با دسته ۳۲: ${digits.training.toLocaleString('en')} در هر تکرار در برابر ${digits.inference.toLocaleString('en')} برای هر پیش‌بینی.`,
        )}
      </P>
      {s.mode === 'infer' ? (
        <>
          <h4>{tr(bi('What runs now', 'چیزی که الان اجرا می‌شود'))}</h4>
          <Tex display>{`\\cyhat{\\hat y} = \\cw{w}\\,\\cx{x} + \\cb{b} = ${tex(w, 3)}\\cdot${tex(s.x, 2)} + ${texP(b, 3)} = \\cyhat{${tex(w * s.x + b, 4)}}`}</Tex>
          <P className="small muted">{bi('That is all. No $\\partial L/\\partial w$ is computed.', 'همین. هیچ $\\partial L/\\partial w$ حساب نمی‌شود.')}</P>
        </>
      ) : (
        <>
          <h4>{tr(bi(`One training iteration, all ${N} samples`, `یک تکرار آموزش، همه ${N} نمونه`))}</h4>
          <Tex display>{`\\cL{L} = \\tfrac{1}{${N}}\\textstyle\\sum_n (\\cw{w}\\cx{x_n} + \\cb{b} - \\cy{y_n})^2 = \\cL{${tex(der.full.loss, 4)}}`}</Tex>
          <Tex display>{`\\cg{\\tfrac{\\partial L}{\\partial w}} = ${tex(gw, 4)},\\quad \\cg{\\tfrac{\\partial L}{\\partial b}} = ${tex(gb, 4)}`}</Tex>
          <Tex display>{`\\cw{w} \\leftarrow ${tex(w, 4)} - \\clr{${tex(opt.lr, 3)}}\\cdot${texP(gw, 4)} = ${tex(w - opt.lr * gw, 4)}`}</Tex>
          <Tex display>{`\\cb{b} \\leftarrow ${tex(b, 4)} - \\clr{${tex(opt.lr, 3)}}\\cdot${texP(gb, 4)} = ${tex(b - opt.lr * gb, 4)}`}</Tex>
        </>
      )}
    </div>
  );
}

function Bottom() {
  const { tr } = useT();
  return (
    <div className="viz-grid">
      <div>
        <div className="small muted">{tr(bi('Training loss (inference never adds points here)', 'خطای آموزش (استنتاج هیچ نقطه‌ای به اینجا اضافه نمی‌کند)'))}</div>
        <LossHistory height={190} />
      </div>
      <TrainStats />
    </div>
  );
}

export const lesson18: LessonDef = {
  id: ID,
  title: bi('Training vs Inference', 'آموزش در برابر استنتاج'),
  short: bi('Train vs inference', 'آموزش و استنتاج'),
  objectives: [
    bi('Run the full training loop: forward → loss → backward → update', 'حلقه کامل آموزش را اجرا کنی: forward ← خطا ← backward ← به‌روزرسانی'),
    bi('Use a trained model for inference with frozen weights', 'از یک مدل آموزش‌دیده با وزن‌های ثابت برای استنتاج استفاده کنی'),
    bi('Compare what is computed, and what it costs, in each mode', 'مقایسه کنی در هر حالت چه چیزی حساب می‌شود و چقدر هزینه دارد'),
    bi('See how CNNs and Transformers reuse the same principles', 'ببینی CNNها و Transformerها چطور از همین اصول استفاده می‌کنند'),
  ],
  presetId: 'temperature',
  revealSteps: 6,
  Explain,
  Center,
  Right,
  Bottom,
  checkpoint: {
    question: bi('Which statement is true during inference?', 'کدام جمله در هنگام استنتاج درست است؟'),
    choices: [
      { id: 'a', label: bi('The weights are nudged a little for every new input', 'وزن‌ها برای هر ورودی جدید کمی تغییر می‌کنند') },
      { id: 'b', label: bi('Only a forward pass runs, with frozen weights', 'فقط یک forward pass با وزن‌های ثابت اجرا می‌شود') },
      { id: 'c', label: bi('The loss is computed against the target', 'خطا نسبت به هدف حساب می‌شود') },
    ],
    answer: 'b',
    explain: bi(
      'New inputs come without targets, so there is no loss and no gradient. The learned weights are only read. Move the inference slider and watch the training loss history: it does not change.',
      'ورودی‌های جدید هدف ندارند، پس خطا و گرادیانی وجود ندارد. وزن‌های یادگرفته‌شده فقط خوانده می‌شوند. اسلایدر استنتاج را حرکت بده و تاریخچه خطای آموزش را نگاه کن: تغییری نمی‌کند.',
    ),
  },
};
