// Lesson 14: Batch, mini-batch and epoch. Uses the shared lab store (preset "mini-batch":
// 8 examples of y ≈ 2x + 1, a single linear neuron, batch size 2 with shuffling). The next batch,
// per-sample gradients, their mean and the history of batches all come from the engine.

import { useMemo } from 'react';
import type { LessonDef } from './types';
import { bi, useT } from '../i18n';
import { useLab, useLabDerived } from '../state/lab';
import { Plot2D, type PlotLayer } from '../components/Plot2D';
import { Tex } from '../components/Tex';
import { Segmented, Slider, Stat } from '../components/controls';
import { Advanced, Callout, Cycle, P, Predict, Reveal } from '../components/teaching';
import { DataEditor, LossHistory } from '../components/LabPanels';
import { flattenGrads, fmt, loadPreset, predict, tex, texP, type BatchConfig } from '../engine';
import { groupByEpoch, meanGrad, perSampleGrads, updatesPerEpoch } from './math/l14';

const ID = 'l14';

/** Restart training from the preset's starting weights (keeps the user's data). */
function restart(patch: Partial<BatchConfig> = {}) {
  const s = useLab.getState();
  s.setRunning(false);
  s.setBatch(patch); // fresh schedule (epoch 0, iteration 0) and optimizer state
  useLab.getState().setNet(loadPreset('mini-batch').net); // fresh weights and history
}

function finishEpoch() {
  const s = useLab.getState();
  const e = s.train.schedule.epoch;
  for (let i = 0; i < s.data.X.length && useLab.getState().train.schedule.epoch === e; i++) if (!useLab.getState().optimizerStep()) break;
}

function useBatchInfo() {
  const data = useLab((s) => s.data);
  const batch = useLab((s) => s.batch);
  const schedule = useLab((s) => s.train.schedule);
  const N = data.X.length;
  const B = batch.size <= 0 || batch.size >= N ? N : batch.size;
  const per = updatesPerEpoch(N, B);
  const groups = Array.from({ length: per }, (_, g) => schedule.order.slice(g * B, Math.min(N, (g + 1) * B)));
  const current = Math.min(per - 1, Math.floor(schedule.cursor / B));
  return { N, B, per, groups, current, schedule, shuffle: batch.shuffle };
}

function Explain() {
  const { tr } = useT();
  const { N, B, per } = useBatchInfo();
  return (
    <>
      <Cycle active={2} />
      <Reveal at={1}>
        <h3>{tr(bi('1. The vocabulary', '۱. واژه‌ها'))}</h3>
        <table className="tbl">
          <tbody>
            <tr><td><strong>{tr(bi('Sample', 'نمونه (sample)'))}</strong></td><td>{tr(bi('one example: an input x and its target y', 'یک مثال: یک ورودی x و هدف آن y'))}</td></tr>
            <tr><td><strong>{tr(bi('Dataset', 'مجموعه داده (dataset)'))}</strong></td><td>{tr(bi(`all ${N} samples`, `همه ${N} نمونه`))}</td></tr>
            <tr><td><strong>{tr(bi('Batch', 'دسته (batch)'))}</strong></td><td>{tr(bi(`the samples used for one update (here B = ${B})`, `نمونه‌هایی که برای یک به‌روزرسانی استفاده می‌شوند (اینجا B = ${B})`))}</td></tr>
            <tr><td><strong>{tr(bi('Mini-batch', 'مینی‌بچ (mini-batch)'))}</strong></td><td>{tr(bi('a batch smaller than the dataset; B = 1 is plain "stochastic" GD, B = N is full-batch GD', 'دسته‌ای کوچک‌تر از کل داده؛ B = 1 همان GD «تصادفی» (stochastic) است و B = N همان GD کامل (full-batch)'))}</td></tr>
            <tr><td><strong>{tr(bi('Iteration', 'تکرار (iteration)'))}</strong></td><td>{tr(bi('one parameter update, from one batch', 'یک به‌روزرسانی پارامترها با یک دسته'))}</td></tr>
            <tr><td><strong>{tr(bi('Epoch', 'دوره (epoch)'))}</strong></td><td>{tr(bi(`one pass in which every sample is used once: ${per} iterations here`, `یک دور که در آن هر نمونه دقیقاً یک بار استفاده می‌شود: اینجا ${per} تکرار`))}</td></tr>
          </tbody>
        </table>
      </Reveal>
      <Reveal at={2}>
        <Predict
          id={ID + '-per-epoch'}
          question={bi('With 8 samples and batch size 2, how many updates happen in one epoch?', 'با ۸ نمونه و اندازه دسته ۲، در یک epoch چند به‌روزرسانی انجام می‌شود؟')}
          choices={[
            { id: '2', label: bi('2', '۲') },
            { id: '4', label: bi('4', '۴') },
            { id: '8', label: bi('8', '۸') },
          ]}
          answer="4"
          explain={bi(
            '8 / 2 = 4 batches, one update each. Press "Finish epoch" and count the new row in the batch timeline below. With batch size 1 it would be 8 updates; with the full batch, 1.',
            '۸ / ۲ = ۴ دسته و برای هر کدام یک به‌روزرسانی. «تمام کردن epoch» را بزن و ردیف جدید را در خط زمانی دسته‌ها پایین بشمار. با اندازه دسته ۱ می‌شد ۸ به‌روزرسانی و با کل داده، ۱.',
          )}
        />
      </Reveal>
      <Reveal at={3}>
        <h3>{tr(bi('2. One gradient from many samples', '۲. یک گرادیان از چند نمونه'))}</h3>
        <P>
          {bi(
            'Each sample $n$ in the batch gets its own forward pass, its own loss $L_n$ and its own gradient $\\cg{g_n}$ from backprop. The batch gradient is simply their average:',
            'هر نمونه $n$ در دسته، forward pass خودش، خطای خودش $L_n$ و گرادیان خودش $\\cg{g_n}$ را از backprop می‌گیرد. گرادیان دسته فقط میانگین آنهاست:',
          )}
        </P>
        <Tex display>{'\\cg{g_{\\text{batch}}} = \\frac{1}{B}\\sum_{n \\in \\text{batch}} \\cg{g_n}, \\qquad \\theta \\leftarrow \\theta - \\clr{\\eta}\\,\\cg{g_{\\text{batch}}}'}</Tex>
        <P>
          {bi(
            'The right panel shows the per-sample gradients of the next batch and their mean, next to the engine\'s batch gradient. Using the mean (not the sum) keeps the step size similar when you change $B$.',
            'پنل سمت راست گرادیان تک‌تک نمونه‌های دسته بعدی و میانگینشان را کنار گرادیان دسته‌ای موتور نشان می‌دهد. استفاده از میانگین (نه جمع) باعث می‌شود با تغییر $B$ اندازه قدم تقریباً ثابت بماند.',
          )}
        </P>
      </Reveal>
      <Reveal at={4}>
        <h3>{tr(bi('3. The training loop', '۳. حلقه آموزش'))}</h3>
        <P>
          {bi(
            'At the start of each epoch the order is shuffled, so batches differ from epoch to epoch. Then, batch by batch: forward, loss, backward, update. The center panel highlights where the loop is right now.',
            'در شروع هر epoch ترتیب نمونه‌ها بُر خورده (shuffle) می‌شود، پس دسته‌ها از یک epoch به بعدی فرق می‌کنند. بعد دسته به دسته: forward، خطا، backward، به‌روزرسانی. پنل وسط نشان می‌دهد حلقه الان کجاست.',
          )}
        </P>
        <Callout>
          {tr(
            bi(
              'Trade-off. B = 1: cheap, noisy steps (the loss curve zig-zags). Full batch: smooth, exact gradient, but every update must look at all the data. Mini-batches sit in between and let a GPU process the batch in parallel.',
              'مصالحه. B = 1: قدم‌های ارزان ولی پرنوسان (منحنی خطا زیگزاگ می‌شود). کل داده: گرادیان دقیق و نرم، ولی هر به‌روزرسانی باید همه داده را ببیند. مینی‌بچ وسط این دو است و اجازه می‌دهد GPU دسته را موازی پردازش کند.',
            ),
          )}
        </Callout>
      </Reveal>
      <Reveal at={5}>
        <Advanced title={bi('Why a mini-batch gradient is a fair estimate', 'چرا گرادیان مینی‌بچ تخمین منصفانه‌ای است')}>
          <P>
            {bi(
              'The full loss is the mean $L = \\frac1N\\sum_n L_n$, so its gradient is the mean of all $g_n$. If a batch is a uniformly random subset, then $\\mathbb E[g_{\\text{batch}}] = \\nabla L$: the estimate is unbiased. Drawing $B$ of $N$ without replacement, its variance is $\\frac{\\sigma^2}{B}\\cdot\\frac{N-B}{N-1}$, where $\\sigma^2$ is the variance of the per-sample gradients: larger batches mean less noise, and $B = N$ means none.',
              'خطای کل میانگین است: $L = \\frac1N\\sum_n L_n$، پس گرادیانش هم میانگین همه $g_n$هاست. اگر دسته یک زیرمجموعه تصادفی یکنواخت باشد، $\\mathbb E[g_{\\text{batch}}] = \\nabla L$: تخمین نااریب است. با انتخاب $B$ از $N$ بدون جایگذاری، واریانس آن $\\frac{\\sigma^2}{B}\\cdot\\frac{N-B}{N-1}$ است که $\\sigma^2$ واریانس گرادیان‌های تک‌نمونه است: دسته بزرگ‌تر یعنی نویز کمتر و $B = N$ یعنی بدون نویز.',
            )}
          </P>
          <P>
            {bi(
              'If $B$ does not divide $N$, the last batch of an epoch is smaller: add a 9th sample with $B = 2$ and the 5th batch holds a single sample. Shuffling every epoch avoids a fixed batch order that could bias the path.',
              'اگر $N$ بر $B$ بخش‌پذیر نباشد، آخرین دسته هر epoch کوچک‌تر است: یک نمونه نهم با $B = 2$ اضافه کن تا دسته پنجم فقط یک نمونه داشته باشد. بُر زدن در هر epoch جلوی ترتیب ثابتی را می‌گیرد که می‌تواند مسیر را سوگیرانه کند.',
            )}
          </P>
        </Advanced>
      </Reveal>
    </>
  );
}

function Center() {
  const { tr } = useT();
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const der = useLabDerived();
  const { N, B, per, groups, current, schedule, shuffle } = useBatchInfo();
  const xs = data.X.map((x) => x[0]);
  const lo = Math.min(...xs) - 0.6;
  const hi = Math.max(...xs) + 0.6;
  const layers = useMemo<PlotLayer[]>(() => {
    const inBatch = new Set(der.nextIdx);
    const out: PlotLayer[] = [{ kind: 'fn', f: (v) => predict(net, [v])[0], color: 'yhat', width: 2.4 }];
    data.X.forEach((x, n) => {
      const yh = predict(net, x)[0];
      out.push({ kind: 'segment', from: [x[0], data.Y[n][0]], to: [x[0], yh], color: inBatch.has(n) ? 'g' : 'var(--muted)', dash: inBatch.has(n) ? undefined : '3 3', width: inBatch.has(n) ? 2.2 : 1 });
    });
    out.push({ kind: 'points', color: 'g', pts: der.nextIdx.map((n) => ({ x: data.X[n][0], y: data.Y[n][0], r: 11, hollow: true })) });
    out.push({ kind: 'points', color: 'y', pts: data.X.map((x, n) => ({ x: x[0], y: data.Y[n][0], r: 5, label: `#${n + 1}` })) });
    return out;
  }, [net, data, der.nextIdx]);
  const lines: { text: string; on: boolean; note?: string }[] = [
    { text: 'for epoch = 1, 2, 3, …', on: false, note: tr(bi(`now: epoch ${schedule.epoch + 1}`, `الان: epoch ${schedule.epoch + 1}`)) },
    { text: '    shuffle(dataset)', on: schedule.cursor === 0 && shuffle, note: shuffle ? tr(bi('on', 'روشن')) : tr(bi('off: fixed order', 'خاموش: ترتیب ثابت')) },
    { text: `    for each batch of ${B}:`, on: false, note: tr(bi(`now: batch ${current + 1} of ${per}`, `الان: دسته ${current + 1} از ${per}`)) },
    { text: '        ŷ = forward(x)', on: true },
    { text: '        L = mean((ŷ − y)²)', on: true },
    { text: '        g = backward(L)', on: true },
    { text: '        θ = θ − η · g', on: true },
  ];
  return (
    <div className="viz">
      <div className="legend-chips">
        <span className="chip"><i style={{ background: 'var(--c-y)' }} /> {tr(bi('sample (x, y)', 'نمونه (x, y)'))}</span>
        <span className="chip"><i style={{ background: 'var(--c-g)' }} /> {tr(bi('in the next batch', 'در دسته بعدی'))}</span>
        <span className="chip"><i style={{ background: 'var(--c-yhat)' }} /> {tr(bi('model ŷ = w·x + b', 'مدل ŷ = w·x + b'))}</span>
      </div>
      <Plot2D testId="batch-scatter" xDomain={[lo, hi]} yDomain="auto" layers={layers} height={300} xLabel="x" yLabel="y, ŷ" />
      <div>
        <div className="small muted">{tr(bi(`This epoch's order, cut into ${per} batches of ${B}:`, `ترتیب این epoch، بریده‌شده به ${per} دسته ${B}تایی:`))}</div>
        <div className="row" style={{ marginTop: 6 }} data-testid="epoch-order">
          {groups.map((g, k) => (
            <span
              key={k}
              className="chip"
              style={{
                opacity: k < current ? 0.4 : 1,
                borderColor: k === current ? 'var(--c-g)' : undefined,
                boxShadow: k === current ? '0 0 0 2px color-mix(in srgb, var(--c-g) 25%, transparent)' : undefined,
                fontWeight: k === current ? 600 : undefined,
              }}
            >
              {g.map((n) => `#${n + 1}`).join(' ')}
            </span>
          ))}
        </div>
      </div>
      <pre className="card flat mono small" style={{ margin: 0, lineHeight: 1.7, overflowX: 'auto' }} dir="ltr" data-testid="loop-code">
        {lines.map((l, i) => (
          <div key={i} style={{ display: 'flex', justifyContent: 'space-between', gap: 16 }}>
            <span style={{ color: l.on ? 'var(--c-g)' : undefined }}>{l.text}</span>
            {l.note && <span className="muted">{l.note}</span>}
          </div>
        ))}
      </pre>
      <p className="small muted">{tr(bi(`N = ${N} samples. Thick magenta lines: the errors that the next update will average.`, `N = ${N} نمونه. خط‌های ضخیم بنفش: خطاهایی که به‌روزرسانی بعدی میانگین می‌گیرد.`))}</p>
    </div>
  );
}

function Right() {
  const { tr } = useT();
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const loss = useLab((s) => s.loss);
  const lossOpts = useLab((s) => s.lossOpts);
  const opt = useLab((s) => s.opt);
  const running = useLab((s) => s.running);
  const setOpt = useLab((s) => s.setOpt);
  const setRunning = useLab((s) => s.setRunning);
  const optimizerStep = useLab((s) => s.optimizerStep);
  const batch = useLab((s) => s.batch);
  const der = useLabDerived();
  const { N, B, per, schedule } = useBatchInfo();
  const rows = useMemo(() => perSampleGrads(net, data, loss, lossOpts, der.nextIdx), [net, data, loss, lossOpts, der.nextIdx]);
  const mean = meanGrad(rows);
  const engine = flattenGrads(net, der.next.grads);
  const full = flattenGrads(net, der.full.grads);
  const w = net.layers[0].W[0][0];
  const b = net.layers[0].b[0];
  const sizeValue = batch.size <= 0 || batch.size >= N ? 0 : batch.size;
  return (
    <div className="stack">
      <div className="row">
        <span className="small">{tr(bi('Batch size', 'اندازه دسته'))}</span>
        <Segmented<number>
          label="batch size"
          value={sizeValue}
          onChange={(size) => restart({ size })}
          options={[1, 2, 4].filter((v) => v < N).map((v) => ({ value: v, label: String(v) })).concat([{ value: 0, label: tr(bi(`full (${N})`, `کل (${N})`)) }])}
        />
      </div>
      <label className="small row" style={{ gap: 6 }}>
        <input type="checkbox" checked={batch.shuffle} onChange={(e) => restart({ shuffle: e.target.checked })} data-testid="shuffle" />
        {tr(bi('Shuffle every epoch', 'بُر زدن در هر epoch'))}
      </label>
      <Slider label={<Tex>{'\\clr{\\eta}'}</Tex>} value={opt.lr} min={0.005} max={0.3} step={0.005} color="lr" onChange={(lr) => setOpt({ lr })} testId="mb-lr" />
      <div className="row">
        <button className="btn" onClick={() => optimizerStep()} disabled={running} data-testid="mb-step">⟳ {tr(bi('One update', 'یک به‌روزرسانی'))}</button>
        <button className="btn" onClick={finishEpoch} disabled={running} data-testid="mb-epoch">⏭ {tr(bi('Finish epoch', 'تمام کردن epoch'))}</button>
        <button className={`btn ${running ? '' : 'primary'}`} onClick={() => setRunning(!running)} data-testid="mb-train">
          {running ? `❚❚ ${tr(bi('Pause', 'توقف'))}` : `▶ ${tr(bi('Train', 'آموزش'))}`}
        </button>
        <button className="btn ghost" onClick={() => restart()} data-testid="mb-reset">↺</button>
      </div>
      <div className="stat-grid">
        <Stat k={tr(bi('iteration', 'تکرار'))} v={schedule.iteration} testId="mb-iter" />
        <Stat k="epoch" v={schedule.epoch} testId="mb-epoch-count" />
        <Stat k={tr(bi('updates / epoch', 'به‌روزرسانی در هر epoch'))} v={`⌈${N}/${B}⌉ = ${per}`} />
        <Stat k={tr(bi('full loss', 'خطای کل'))} v={fmt(der.full.loss, 4)} color="L" />
      </div>
      <h4>{tr(bi('Next batch: per-sample gradients', 'دسته بعدی: گرادیان هر نمونه'))}</h4>
      <div style={{ maxHeight: 220, overflow: 'auto' }}>
        <table className="tbl" data-testid="grad-table">
          <thead>
            <tr>
              <th>#</th>
              <th className="num c-x">x</th>
              <th className="num c-y">y</th>
              <th className="num c-yhat">ŷ</th>
              <th className="num c-g">∂Lₙ/∂w</th>
              <th className="num c-g">∂Lₙ/∂b</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.n}>
                <td>{r.n + 1}</td>
                <td className="num">{fmt(r.x[0], 3)}</td>
                <td className="num">{fmt(r.y[0], 3)}</td>
                <td className="num">{fmt(r.yhat[0], 3)}</td>
                <td className="num">{fmt(r.g[0], 4)}</td>
                <td className="num">{fmt(r.g[1], 4)}</td>
              </tr>
            ))}
            <tr className="hl">
              <td colSpan={4}>{tr(bi(`mean of ${rows.length}`, `میانگین ${rows.length} نمونه`))}</td>
              <td className="num c-g">{fmt(mean[0], 4)}</td>
              <td className="num c-g">{fmt(mean[1], 4)}</td>
            </tr>
            <tr>
              <td colSpan={4} className="muted">{tr(bi('engine batch gradient', 'گرادیان دسته‌ای موتور'))}</td>
              <td className="num">{fmt(engine[0], 4)}</td>
              <td className="num">{fmt(engine[1], 4)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <Tex display>{`\\cg{g_w} = \\tfrac{1}{${rows.length}}(${rows.map((r) => texP(r.g[0], 3)).join(' + ')}) = \\cg{${tex(mean[0], 4)}}`}</Tex>
      <Tex display>{`\\cw{w} \\leftarrow ${tex(w, 4)} - \\clr{${tex(opt.lr, 3)}}\\cdot${texP(mean[0], 4)} = ${tex(w - opt.lr * mean[0], 4)}`}</Tex>
      <Tex display>{`\\cb{b} \\leftarrow ${tex(b, 4)} - \\clr{${tex(opt.lr, 3)}}\\cdot${texP(mean[1], 4)} = ${tex(b - opt.lr * mean[1], 4)}`}</Tex>
      <P className="small muted">
        {bi(
          `For comparison, the full-dataset gradient is (${fmt(full[0], 4)}, ${fmt(full[1], 4)}). The batch gradient points roughly the same way but is noisier.`,
          `برای مقایسه، گرادیان کل داده (${fmt(full[0], 4)}, ${fmt(full[1], 4)}) است. گرادیان دسته تقریباً همان جهت را نشان می‌دهد ولی پرنویزتر است.`,
        )}
      </P>
    </div>
  );
}

function Timeline() {
  const { tr } = useT();
  const history = useLab((s) => s.history);
  const groups = useMemo(() => groupByEpoch(history).slice(-6), [history]);
  if (!groups.length) return <P className="small muted">{bi('No updates yet. Press "One update".', 'هنوز به‌روزرسانی انجام نشده. «یک به‌روزرسانی» را بزن.')}</P>;
  return (
    <div className="stack" style={{ gap: 6 }} data-testid="batch-timeline">
      {groups.map((g) => (
        <div key={g.epoch} className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
          <span className="small muted" style={{ minWidth: '5.5em' }}>
            epoch {g.epoch + 1}
          </span>
          {g.steps.map((st) => (
            <span key={st.iter} className="chip" title={tr(bi(`iteration ${st.iter}`, `تکرار ${st.iter}`))}>
              <span className="muted">{st.iter}:</span> {st.batch.map((n) => `#${n + 1}`).join(' ')}
            </span>
          ))}
        </div>
      ))}
    </div>
  );
}

function Bottom() {
  const { tr } = useT();
  return (
    <div className="viz-grid">
      <div>
        <div className="small muted">{tr(bi('Dataset (• = in the next batch). Edit x or y to change a sample.', 'مجموعه داده (• = در دسته بعدی). x یا y را ویرایش کن تا نمونه عوض شود.'))}</div>
        <DataEditor />
      </div>
      <div>
        <div className="small muted">{tr(bi('Batches used so far, grouped by epoch (iteration: samples)', 'دسته‌های استفاده‌شده تا حالا، به تفکیک epoch (تکرار: نمونه‌ها)'))}</div>
        <Timeline />
      </div>
      <div>
        <div className="small muted">{tr(bi('Loss on the whole dataset after each update', 'خطای کل داده بعد از هر به‌روزرسانی'))}</div>
        <LossHistory height={200} />
      </div>
    </div>
  );
}

export const lesson14: LessonDef = {
  id: ID,
  title: bi('Batch, Mini-Batch, and Epoch', 'دسته، مینی‌بچ و دوره (Epoch)'),
  short: bi('Batches & epochs', 'دسته‌ها و epoch'),
  objectives: [
    bi('Tell apart sample, dataset, batch, mini-batch, iteration and epoch', 'نمونه، مجموعه داده، دسته، مینی‌بچ، تکرار و epoch را از هم تشخیص بدهی'),
    bi('Compute a batch gradient as the mean of per-sample gradients', 'گرادیان دسته را به شکل میانگین گرادیان نمونه‌ها حساب کنی'),
    bi('Count updates per epoch for a given batch size', 'تعداد به‌روزرسانی‌ها در هر epoch را برای یک اندازه دسته بشماری'),
    bi('Follow the loop: shuffle, then forward, loss, backward, update per batch', 'حلقه را دنبال کنی: بُر زدن، بعد برای هر دسته forward، خطا، backward، به‌روزرسانی'),
  ],
  presetId: 'mini-batch',
  revealSteps: 5,
  Explain,
  Center,
  Right,
  Bottom,
  checkpoint: {
    question: bi('A dataset has 8 samples and the batch size is 4. After 6 updates, how many epochs are complete?', 'یک مجموعه داده ۸ نمونه دارد و اندازه دسته ۴ است. بعد از ۶ به‌روزرسانی چند epoch کامل شده است؟'),
    choices: [
      { id: 'a', label: bi('1.5', '۱٫۵') },
      { id: 'b', label: bi('3', '۳') },
      { id: 'c', label: bi('6', '۶') },
      { id: 'd', label: bi('24', '۲۴') },
    ],
    answer: 'b',
    explain: bi(
      'One epoch is 8 / 4 = 2 updates, so 6 updates make 3 epochs. Set the batch size to 4, press "One update" six times and read the epoch counter.',
      'هر epoch برابر ۸ / ۴ = ۲ به‌روزرسانی است، پس ۶ به‌روزرسانی یعنی ۳ epoch. اندازه دسته را ۴ بگذار، شش بار «یک به‌روزرسانی» را بزن و شمارنده epoch را بخوان.',
    ),
  },
};
