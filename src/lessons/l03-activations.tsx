// Lesson 03: Activation functions. Stacked affine maps collapse into one affine map; an activation
// between them breaks the collapse. Then ReLU, sigmoid, tanh and linear with their derivatives,
// all from the engine (act / actDeriv / predict). Lesson-local state.

import { useMemo } from 'react';
import type { LessonDef } from './types';
import { bi, useT, type Bi } from '../i18n';
import { useLessonState } from '../state/lessonState';
import { Plot2D, type PlotLayer } from '../components/Plot2D';
import { Tex } from '../components/Tex';
import { Segmented, Slider, Stat } from '../components/controls';
import { Advanced, Callout, Cycle, P, Predict, Reveal } from '../components/teaching';
import { act, actDeriv, clamp, fmt, fmtSigned, predict, tex, texP, type Network } from '../engine';

const ID = 'l03';

type Act = 'relu' | 'sigmoid' | 'tanh' | 'linear';
const ACTS: Act[] = ['relu', 'sigmoid', 'tanh', 'linear'];

interface S {
  view: 'stack' | 'act';
  act: Act;
  z: number;
  x: number;
  w1: number;
  b1: number;
  w2: number;
  b2: number;
  mid: 'linear' | 'relu';
}
const INIT: S = { view: 'stack', act: 'relu', z: 1, x: 1, w1: 1.5, b1: -1, w2: 2, b2: 0.5, mid: 'linear' };
const useS = () => useLessonState<S>(ID, INIT);

const ACT_TEX: Record<Act, { f: string; d: string; name: Bi }> = {
  relu: {
    f: '\\relu(z) = \\max(0, z)',
    d: "\\relu'(z) = \\begin{cases} 0 & z < 0 \\\\ \\text{undefined} & z = 0 \\\\ 1 & z > 0 \\end{cases}",
    name: bi('ReLU', 'ReLU'),
  },
  sigmoid: { f: '\\sigma(z) = \\dfrac{1}{1 + e^{-z}}', d: "\\sigma'(z) = \\sigma(z)\\,\\big(1 - \\sigma(z)\\big)", name: bi('Sigmoid', 'سیگموید (Sigmoid)') },
  tanh: { f: '\\tanh(z) = \\dfrac{e^{z} - e^{-z}}{e^{z} + e^{-z}}', d: "\\tanh'(z) = 1 - \\tanh^2(z)", name: bi('Tanh', 'تانژانت هذلولوی (tanh)') },
  linear: { f: 'f(z) = z', d: "f'(z) = 1", name: bi('Linear (identity)', 'خطی (همانی)') },
};
const Y_DOMAIN: Record<Act, [number, number]> = { relu: [-1, 4], sigmoid: [-0.25, 1.25], tanh: [-1.3, 1.3], linear: [-4, 4] };

/** The two-layer chain x -> w1 x + b1 -> f -> w2 (.) + b2 as an engine network. */
const twoLayer = (s: S): Network => ({
  inputSize: 1,
  layers: [
    { W: [[s.w1]], b: [s.b1], act: s.mid },
    { W: [[s.w2]], b: [s.b2], act: 'linear' },
  ],
});
/** The single affine map W x + B that two affine layers collapse into. */
const collapsed = (s: S): Network => ({ inputSize: 1, layers: [{ W: [[s.w2 * s.w1]], b: [s.w2 * s.b1 + s.b2], act: 'linear' }] });
const clean = (v: number) => (Math.abs(v) < 1e-9 ? 0 : v);

function Explain() {
  const { tr } = useT();
  const [s, set] = useS();
  return (
    <>
      <Cycle active={s.view === 'stack' ? 1 : 3} />
      <Reveal at={1}>
        <h3>{tr(bi('1. Stacking lines gives a line', '۱. خط روی خط، باز هم خط'))}</h3>
        <P>
          {bi(
            'Feed one neuron into another, with nothing in between: $\\hat y = w_2(w_1 x + b_1) + b_2$. Multiply it out and it is just one line again:',
            'یک نورون را مستقیم به نورون دیگری بده، بدون هیچ چیز در میان: $\\hat y = w_2(w_1 x + b_1) + b_2$. اگر پرانتز را باز کنی، باز فقط یک خط است:',
          )}
        </P>
        <Tex display>{'w_2(w_1 x + b_1) + b_2 = \\underbrace{(w_2 w_1)}_{W}\\,x + \\underbrace{(w_2 b_1 + b_2)}_{B}'}</Tex>
        <P>
          {bi(
            'Ten such layers, or a hundred, still collapse into one $Wx + B$. But real relationships bend: the fan from lesson 1 is flat, then rises, then is flat again. Lines alone can never draw that.',
            'ده لایه یا صد لایه از این نوع هم باز در یک $Wx + B$ خلاصه می‌شوند. اما رابطه‌های واقعی خم دارند: پنکه درس ۱ اول صاف است، بعد بالا می‌رود و دوباره صاف می‌شود. فقط با خط هیچ‌وقت نمی‌شود چنین چیزی کشید.',
          )}
        </P>
      </Reveal>
      <Reveal at={2}>
        <Predict
          id={ID + '-bend'}
          question={bi('With no activation between the layers, can changing $w_1, b_1, w_2, b_2$ ever make the output bend?', 'بدون تابع فعال‌سازی بین لایه‌ها، آیا با تغییر $w_1, b_1, w_2, b_2$ می‌شود خروجی را خم کرد؟')}
          choices={[
            { id: 'yes', label: bi('Yes, with the right numbers', 'بله، با عددهای مناسب') },
            { id: 'no', label: bi('No, it is always a straight line', 'نه، همیشه یک خط راست است') },
          ]}
          answer="no"
          explain={bi(
            'Move any slider: the solid two-layer curve always sits exactly on the dashed one-layer line $Wx + B$, and the table below shows a difference of 0 everywhere. Now switch "between the layers" to ReLU.',
            'هر اسلایدری را تکان بده: منحنی دولایه همیشه دقیقاً روی خط‌چین تک‌لایه $Wx + B$ است و جدول پایین همه‌جا اختلاف ۰ نشان می‌دهد. حالا «بین لایه‌ها» را روی ReLU بگذار.',
          )}
        />
      </Reveal>
      <Reveal at={3}>
        <h3>{tr(bi('2. The activation function', '۲. تابع فعال‌سازی'))}</h3>
        <P>
          {bi(
            'The fix is a small nonlinear function applied to each neuron\'s $\\cz{z}$. Its result is the neuron\'s activation $\\ca{a} = f(\\cz{z})$. The most popular one is ReLU:',
            'راه‌حل یک تابع غیرخطی کوچک است که روی $\\cz{z}$ هر نورون اعمال می‌شود. نتیجه‌اش خروجی فعال‌شده نورون است: $\\ca{a} = f(\\cz{z})$. محبوب‌ترین آنها ReLU است:',
          )}
        </P>
        <Tex display>{'\\ca{a} = \\relu(\\cz{z}) = \\max(0, \\cz{z})'}</Tex>
        <P>
          {bi(
            'Negative $z$: output 0 (the neuron is "off"). Positive $z$: output $z$ unchanged ("on"). Exactly at $z = 0$ both rules give 0. With ReLU between the layers the output bends at the input where $z_1 = 0$, which is $x = -b_1/w_1$.',
            '$z$ منفی: خروجی ۰ (نورون «خاموش» است). $z$ مثبت: خروجی همان $z$ («روشن»). دقیقاً در $z = 0$ هر دو قاعده ۰ می‌دهند. با ReLU بین لایه‌ها، خروجی در ورودی‌ای خم می‌شود که در آن $z_1 = 0$ است، یعنی $x = -b_1/w_1$.',
          )}
        </P>
        {s.view !== 'act' && (
          <button className="btn sm" onClick={() => set({ view: 'act', act: 'relu' })} data-testid="to-act">
            {tr(bi('Explore activation functions →', 'توابع فعال‌سازی را ببین ←'))}
          </button>
        )}
      </Reveal>
      <Reveal at={4}>
        <Predict
          id={ID + '-relu-dir'}
          question={bi('$z$ increases from $-3$ to $-1$. What does $\\relu(z)$ do?', '$z$ از $-3$ به $-1$ زیاد می‌شود. $\\relu(z)$ چه می‌کند؟')}
          choices={[
            { id: 'up', label: bi('Increases', 'زیاد می‌شود') },
            { id: 'flat', label: bi('Stays at 0', 'روی ۰ می‌ماند') },
            { id: 'down', label: bi('Decreases', 'کم می‌شود') },
          ]}
          answer="flat"
          explain={bi(
            'Everything below 0 is cut to 0, so small changes there change nothing: the slope is 0. Drag $z$ across 0 and watch the slope jump from 0 to 1.',
            'هر چیزی زیر ۰ به ۰ بریده می‌شود، پس تغییرهای کوچک آنجا هیچ اثری ندارند: شیب ۰ است. $z$ را از روی ۰ عبور بده و ببین شیب چطور از ۰ به ۱ می‌پرد.',
          )}
        />
        <h3>{tr(bi('3. ReLU\'s slope', '۳. شیب ReLU'))}</h3>
        <P>
          {bi(
            'Left of 0 the slope is 0; right of 0 it is 1. At exactly $z = 0$ the slope from the left (0) and from the right (1) disagree, so the derivative is undefined there. Training code still needs a number, so it picks a convention: this studio, like PyTorch, uses $\\relu\'(0) = 0$.',
            'سمت چپ ۰ شیب ۰ است و سمت راست ۱. دقیقاً در $z = 0$ شیب از چپ (۰) و از راست (۱) با هم نمی‌خوانند، پس مشتق آنجا تعریف نشده است. کد آموزش به هر حال یک عدد لازم دارد و یک قرارداد انتخاب می‌کند: این استودیو هم مثل PyTorch از $\\relu\'(0) = 0$ استفاده می‌کند.',
          )}
        </P>
      </Reveal>
      <Reveal at={5}>
        <h3>{tr(bi('4. Other activations', '۴. توابع فعال‌سازی دیگر'))}</h3>
        <P>
          {bi(
            'Sigmoid squashes any $z$ into $(0, 1)$, handy for probabilities. Tanh squashes into $(-1, 1)$ and is centred at 0. Linear (identity) changes nothing; it is used on the output of a regression network. Far from 0, sigmoid and tanh become flat ("saturate") and their slopes approach 0.',
            'سیگموید هر $z$ را به بازه $(0, 1)$ فشرده می‌کند و برای احتمال مناسب است. tanh آن را به $(-1, 1)$ فشرده می‌کند و مرکزش ۰ است. خطی (همانی) چیزی را عوض نمی‌کند و در خروجی شبکه‌های رگرسیون به کار می‌رود. دور از ۰، سیگموید و tanh صاف می‌شوند («اشباع» یا saturation) و شیبشان به ۰ نزدیک می‌شود.',
          )}
        </P>
        <Callout tone="warn">
          {tr(
            bi(
              'Looking ahead: if a ReLU neuron\'s z is negative for every training example, its output and its slope are 0 everywhere, so no learning signal reaches its weights. This is a "dead ReLU". We will meet it again when we train networks.',
              'نگاهی به جلو: اگر z یک نورون ReLU برای همه مثال‌های آموزشی منفی باشد، خروجی و شیبش همه‌جا ۰ است و هیچ سیگنال یادگیری‌ای به وزن‌هایش نمی‌رسد. به این «ReLU مرده» (dead ReLU) می‌گویند. وقتی شبکه‌ها را آموزش بدهیم دوباره با آن روبه‌رو می‌شویم.',
            ),
          )}
        </Callout>
      </Reveal>
      <Reveal at={6}>
        <Advanced>
          <P>
            {bi(
              'In matrix form, two affine layers give $W_2(W_1\\mathbf{x} + \\mathbf{b}_1) + \\mathbf{b}_2 = (W_2W_1)\\mathbf{x} + (W_2\\mathbf{b}_1 + \\mathbf{b}_2)$: affine maps are closed under composition, so depth without nonlinearity adds no expressive power.',
              'به شکل ماتریسی، دو لایه آفین می‌شوند $W_2(W_1\\mathbf{x} + \\mathbf{b}_1) + \\mathbf{b}_2 = (W_2W_1)\\mathbf{x} + (W_2\\mathbf{b}_1 + \\mathbf{b}_2)$: ترکیب نگاشت‌های آفین باز آفین است، پس عمق بدون غیرخطی بودن هیچ قدرت بیان اضافه‌ای نمی‌دهد.',
            )}
          </P>
          <P>
            {bi(
              'With a nonlinearity, one hidden layer of enough neurons can approximate any continuous function on a bounded interval as closely as you like (universal approximation). ReLU networks represent exactly the continuous piecewise-linear functions.',
              'با یک تابع غیرخطی، یک لایه پنهان با نورون‌های کافی می‌تواند هر تابع پیوسته روی یک بازه کراندار را با هر دقتی تقریب بزند (تقریب‌زنی جهانی یا universal approximation). شبکه‌های ReLU دقیقاً تابع‌های پیوسته تکه‌ای‌خطی را نمایش می‌دهند.',
            )}
          </P>
          <P>
            {bi(
              'At $z = 0$ every value in $[0, 1]$ is a valid subgradient of ReLU; picking 0 is a choice, not a fact. Useful identities: $\\tanh(z) = 2\\sigma(2z) - 1$, $\\max \\sigma\' = \\sigma\'(0) = 1/4$, $\\max \\tanh\' = \\tanh\'(0) = 1$.',
              'در $z = 0$ هر مقدار در $[0, 1]$ یک زیرگرادیان (subgradient) معتبر برای ReLU است؛ انتخاب ۰ یک تصمیم است، نه یک واقعیت ریاضی. چند رابطه مفید: $\\tanh(z) = 2\\sigma(2z) - 1$، $\\max \\sigma\' = \\sigma\'(0) = 1/4$ و $\\max \\tanh\' = \\tanh\'(0) = 1$.',
            )}
          </P>
        </Advanced>
      </Reveal>
    </>
  );
}

function actLayers(a: Act, z: number, small = false): PlotLayer[] {
  const out: PlotLayer[] = [];
  if (a === 'relu' && !small) {
    out.push(
      { kind: 'band', x0: -10, x1: 0, color: 'var(--c-wneg)', opacity: 0.07, label: 'z < 0 → 0' },
      { kind: 'band', x0: 0, x1: 10, color: 'var(--c-wpos)', opacity: 0.07, label: 'z > 0 → z' },
    );
  }
  out.push({ kind: 'fn', f: (t) => act(a, t), color: 'a', width: small ? 2 : 2.8 });
  if (a === 'relu') {
    // Draw the jump honestly: two flat pieces, open circle at (0, 1), filled at the convention (0, 0).
    out.push(
      { kind: 'segment', from: [-10, 0], to: [0, 0], color: 'g', dash: '6 4', width: 1.8 },
      { kind: 'segment', from: [0, 1], to: [10, 1], color: 'g', dash: '6 4', width: 1.8 },
      { kind: 'points', color: 'g', pts: [{ x: 0, y: 1, hollow: true, r: 4 }, { x: 0, y: 0, r: 3.5, label: small ? undefined : "ReLU′(0) := 0" }] },
    );
  } else {
    out.push({ kind: 'fn', f: (t) => actDeriv(a, t), color: 'g', dash: '6 4', width: 1.8 });
  }
  if (!small) out.push({ kind: 'line', p: [z, act(a, z)], slope: actDeriv(a, z), color: 'g', width: 1.4, opacity: 0.8 });
  out.push({ kind: 'points', color: 'a', pts: [{ x: z, y: act(a, z), r: small ? 3.5 : 5 }] });
  return out;
}

function Center() {
  const { tr } = useT();
  const [s, set] = useS();
  const net = twoLayer(s);
  const one = collapsed(s);
  const yhat = predict(net, [s.x])[0];
  const z1 = s.w1 * s.x + s.b1;
  const h = act(s.mid, z1);
  const layers = useMemo<PlotLayer[]>(() => {
    if (s.view === 'act') return actLayers(s.act, s.z);
    const out: PlotLayer[] = [
      { kind: 'fn', f: (t) => predict(twoLayer(s), [t])[0], color: 'yhat', width: 3.4 },
      { kind: 'fn', f: (t) => predict(collapsed(s), [t])[0], color: 'z', width: 1.8, dash: '7 5' },
    ];
    if (s.mid === 'relu' && Math.abs(s.w1) > 1e-9) out.push({ kind: 'vline', at: -s.b1 / s.w1, color: 'a', label: 'z₁ = 0' });
    return out;
  }, [s]);
  return (
    <div className="viz">
      <div className="row">
        <Segmented
          label="view"
          value={s.view}
          onChange={(view) => set({ view })}
          options={[
            { value: 'stack', label: tr(bi('Stacking layers', 'روی هم گذاشتن لایه‌ها')) },
            { value: 'act', label: tr(bi('Activation functions', 'توابع فعال‌سازی')) },
          ]}
        />
        {s.view === 'stack' ? (
          <Segmented
            label="between"
            value={s.mid}
            onChange={(mid) => set({ mid })}
            options={[
              { value: 'linear', label: tr(bi('Nothing between', 'هیچ چیز بین لایه‌ها')) },
              { value: 'relu', label: tr(bi('ReLU between', 'ReLU بین لایه‌ها')) },
            ]}
          />
        ) : (
          <Segmented label="activation" value={s.act} onChange={(a) => set({ act: a })} options={ACTS.map((a) => ({ value: a, label: tr(ACT_TEX[a].name) }))} />
        )}
      </div>
      {s.view === 'stack' ? (
        <>
          <div className="formula-hero">
            <Tex>{`\\cx{${tex(s.x, 2)}} \\xrightarrow{\\;w_1 x + b_1\\;} \\cz{${tex(z1, 3)}} \\xrightarrow{\\;${s.mid === 'relu' ? '\\relu' : '\\text{id}'}\\;} \\ca{${tex(h, 3)}} \\xrightarrow{\\;w_2 a + b_2\\;} \\cyhat{${tex(yhat, 3)}}`}</Tex>
          </div>
          <Plot2D
            testId="stack-plot"
            xDomain={[-4, 4]}
            yDomain={[-8, 8]}
            layers={layers}
            height={320}
            xLabel={tr(bi('input x', 'ورودی x'))}
            yLabel="ŷ"
            handle={{ x: s.x, y: yhat, color: 'yhat' }}
            onDrag={(x) => set({ x: clamp(Math.round(x * 10) / 10, -4, 4) })}
            legend={[
              { label: tr(bi('two layers', 'دو لایه')), color: 'yhat' },
              { label: `${tr(bi('one layer', 'یک لایه'))}: ${fmt(one.layers[0].W[0][0], 3)}·x + ${fmt(one.layers[0].b[0], 3)}`, color: 'z', dash: true },
            ]}
            ariaLabel="two stacked layers versus one layer"
          />
          <Callout tone={s.mid === 'relu' ? 'warn' : 'info'}>
            {s.mid === 'relu'
              ? tr(bi('With ReLU between, the output bends where z₁ = 0 and leaves the dashed line: no single line can match it.', 'با ReLU بین لایه‌ها، خروجی جایی که z₁ = 0 است خم می‌شود و از خط‌چین جدا می‌شود: هیچ خط واحدی نمی‌تواند با آن یکی شود.'))
              : tr(bi('Without an activation the two-layer output lies exactly on one line, whatever the sliders say.', 'بدون تابع فعال‌سازی، خروجی دولایه دقیقاً روی یک خط است، هر عددی که اسلایدرها نشان بدهند.'))}
          </Callout>
        </>
      ) : (
        <>
          <Plot2D
            testId="act-plot"
            xDomain={[-4, 4]}
            yDomain={Y_DOMAIN[s.act]}
            layers={layers}
            height={340}
            xLabel="z"
            yLabel={tr(bi('a = f(z) and slope f′(z)', 'a = f(z) و شیب f′(z)'))}
            handle={{ x: s.z, y: act(s.act, s.z), color: 'a' }}
            onDrag={(x) => set({ z: clamp(Math.round(x * 20) / 20, -4, 4) })}
            legend={[
              { label: 'f(z)', color: 'a' },
              { label: "f′(z)", color: 'g', dash: true },
              { label: tr(bi('tangent at z', 'مماس در z')), color: 'g' },
            ]}
            ariaLabel="activation function and its derivative"
          />
          <p className="small muted">{tr(bi('Drag on the plot to move z.', 'روی نمودار بکش تا z جابه‌جا شود.'))}</p>
        </>
      )}
    </div>
  );
}

function Right() {
  const { tr } = useT();
  const [s, set] = useS();
  if (s.view === 'stack') {
    const W = s.w2 * s.w1;
    const B = s.w2 * s.b1 + s.b2;
    const two = predict(twoLayer(s), [s.x])[0];
    const one = predict(collapsed(s), [s.x])[0];
    return (
      <div className="stack">
        <Slider label="x" value={s.x} min={-4} max={4} step={0.1} color="x" onChange={(x) => set({ x })} />
        <Slider label="w₁" value={s.w1} min={-3} max={3} step={0.05} color="w" onChange={(w1) => set({ w1 })} testId="l03-w1" />
        <Slider label="b₁" value={s.b1} min={-4} max={4} step={0.1} color="b" onChange={(b1) => set({ b1 })} />
        <Slider label="w₂" value={s.w2} min={-3} max={3} step={0.05} color="w" onChange={(w2) => set({ w2 })} />
        <Slider label="b₂" value={s.b2} min={-4} max={4} step={0.1} color="b" onChange={(b2) => set({ b2 })} />
        {s.mid === 'linear' ? (
          <>
            <Tex display>{`\\cyhat{\\hat y} = ${texP(s.w2, 2)}\\big(${texP(s.w1, 2)}\\,x + ${texP(s.b1, 2)}\\big) + ${texP(s.b2, 2)}`}</Tex>
            <Tex display>{`= (${texP(s.w2, 2)}\\cdot${texP(s.w1, 2)})\\,x + (${texP(s.w2, 2)}\\cdot${texP(s.b1, 2)} + ${texP(s.b2, 2)}) = ${tex(W, 3)}\\,x + ${texP(B, 3)}`}</Tex>
          </>
        ) : (
          <>
            <Tex display>{`\\cyhat{\\hat y} = ${texP(s.w2, 2)}\\,\\relu\\big(${texP(s.w1, 2)}\\,x + ${texP(s.b1, 2)}\\big) + ${texP(s.b2, 2)}`}</Tex>
            <Tex display>{`= \\begin{cases} ${tex(W, 3)}\\,x + ${texP(B, 3)} & \\text{if } w_1x + b_1 > 0 \\\\ ${tex(s.b2, 3)} & \\text{otherwise} \\end{cases}`}</Tex>
          </>
        )}
        <div className="stat-grid">
          <Stat k={tr(bi('two layers', 'دو لایه'))} v={fmt(two, 4)} color="yhat" testId="l03-two" />
          <Stat k="W·x + B" v={fmt(one, 4)} color="z" testId="l03-one" />
          <Stat k={tr(bi('difference', 'اختلاف'))} v={fmt(clean(two - one), 4)} />
        </div>
      </div>
    );
  }
  const f = (t: number) => act(s.act, t);
  const h = 1e-3;
  const left = (f(s.z) - f(s.z - h)) / h;
  const right = (f(s.z + h) - f(s.z)) / h;
  const atKink = s.act === 'relu' && s.z === 0;
  return (
    <div className="stack">
      <Slider label="z" value={s.z} min={-4} max={4} step={0.05} color="z" onChange={(z) => set({ z })} testId="l03-z" />
      <div className="row">
        <button className="btn sm" onClick={() => set({ z: 0 })} data-testid="z-zero">
          z = 0
        </button>
        <button className="btn sm" onClick={() => set({ z: -2 })}>
          z = −2
        </button>
        <button className="btn sm" onClick={() => set({ z: 2 })}>
          z = 2
        </button>
      </div>
      <Tex display>{ACT_TEX[s.act].f}</Tex>
      <Tex display>{ACT_TEX[s.act].d}</Tex>
      <Tex display>{`\\ca{a} = f(\\cz{${tex(s.z, 3)}}) = \\ca{${tex(f(s.z), 4)}}`}</Tex>
      <div className="stat-grid">
        <Stat k="f(z)" v={fmt(f(s.z), 4)} color="a" testId="l03-fz" />
        <Stat k={tr(bi("f′(z), engine", 'f′(z) موتور'))} v={fmt(actDeriv(s.act, s.z), 4)} color="g" testId="l03-dfz" />
        <Stat k={tr(bi('slope from the left', 'شیب از چپ'))} v={fmt(clean(left), 4)} />
        <Stat k={tr(bi('slope from the right', 'شیب از راست'))} v={fmt(clean(right), 4)} />
      </div>
      <P className="small muted">{bi('One-sided slopes use a step $h = 0.001$: $\\frac{f(z)-f(z-h)}{h}$ and $\\frac{f(z+h)-f(z)}{h}$.', 'شیب‌های یک‌طرفه با قدم $h = 0.001$ حساب شده‌اند: $\\frac{f(z)-f(z-h)}{h}$ و $\\frac{f(z+h)-f(z)}{h}$.')}</P>
      {atKink && (
        <Callout tone="warn">
          {tr(
            bi(
              'At z = 0 the left slope is 0 and the right slope is 1: they disagree, so ReLU′(0) does not exist. The engine (like PyTorch) uses 0 by convention.',
              'در z = 0 شیب از چپ ۰ و از راست ۱ است: با هم نمی‌خوانند، پس ReLU′(0) وجود ندارد. موتور (مثل PyTorch) طبق قرارداد ۰ را استفاده می‌کند.',
            ),
          )}
        </Callout>
      )}
      {(s.act === 'sigmoid' || s.act === 'tanh') && Math.abs(s.z) >= 3 && (
        <Callout>{tr(bi('Saturated: the curve is almost flat here, so its slope is close to 0.', 'اشباع: منحنی اینجا تقریباً صاف است، پس شیبش نزدیک ۰ است.'))}</Callout>
      )}
    </div>
  );
}

function Bottom() {
  const { tr } = useT();
  const [s, set] = useS();
  if (s.view === 'stack') {
    const xs = [-3, -2, -1, 0, 1, 2, 3];
    return (
      <table className="tbl" data-testid="stack-table">
        <thead>
          <tr>
            <th className="num c-x">x</th>
            <th className="num c-z">z₁ = w₁x + b₁</th>
            <th className="num c-a">{s.mid === 'relu' ? 'a₁ = ReLU(z₁)' : 'a₁ = z₁'}</th>
            <th className="num c-yhat">ŷ = w₂a₁ + b₂</th>
            <th className="num">W·x + B</th>
            <th className="num">{tr(bi('difference', 'اختلاف'))}</th>
          </tr>
        </thead>
        <tbody>
          {xs.map((x) => {
            const z1 = s.w1 * x + s.b1;
            const two = predict(twoLayer(s), [x])[0];
            const one = predict(collapsed(s), [x])[0];
            return (
              <tr key={x} className={Math.abs(x - s.x) < 1e-9 ? 'hl' : ''}>
                <td className="num">{x}</td>
                <td className="num">{fmt(z1, 3)}</td>
                <td className="num">{fmt(act(s.mid, z1), 3)}</td>
                <td className="num">{fmt(two, 3)}</td>
                <td className="num">{fmt(one, 3)}</td>
                <td className={`num ${Math.abs(two - one) > 1e-9 ? 'c-L' : ''}`}>{fmtSigned(clean(two - one), 3)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    );
  }
  return (
    <div className="viz-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))' }}>
      {ACTS.map((a) => (
        <div key={a} onClick={() => set({ act: a })} style={{ cursor: 'pointer', outline: a === s.act ? '2px solid var(--accent)' : undefined, borderRadius: 8, padding: 4 }} data-testid={`mini-${a}`}>
          <div className="small">
            <strong>{tr(ACT_TEX[a].name)}</strong>{' '}
            <span className="muted">
              f({fmt(s.z, 2)}) = {fmt(act(a, s.z), 3)}, f′ = {fmt(actDeriv(a, s.z), 3)}
            </span>
          </div>
          <Plot2D xDomain={[-4, 4]} yDomain={[-1.3, 2.3]} height={150} layers={actLayers(a, s.z, true)} ariaLabel={a} />
        </div>
      ))}
    </div>
  );
}

export const lesson03: LessonDef = {
  id: ID,
  title: bi('Activation Functions', 'توابع فعال‌سازی'),
  short: bi('Activations', 'فعال‌سازی'),
  objectives: [
    bi('Show that stacked affine layers collapse into one affine map', 'نشان بدهی لایه‌های آفین پشت سر هم در یک نگاشت آفین خلاصه می‌شوند'),
    bi('Describe ReLU\'s negative, zero and positive regions and how it bends a graph', 'ناحیه‌های منفی، صفر و مثبت ReLU و اینکه چطور نمودار را خم می‌کند توضیح بدهی'),
    bi('Read ReLU\'s derivative, including why it is undefined at 0 and which convention is used', 'مشتق ReLU را بخوانی، از جمله اینکه چرا در ۰ تعریف نشده و چه قراردادی استفاده می‌شود'),
    bi('Compare sigmoid, tanh and linear activations and their slopes', 'سیگموید، tanh و خطی را با شیب‌هایشان مقایسه کنی'),
  ],
  revealSteps: 6,
  Explain,
  Center,
  Right,
  Bottom,
  checkpoint: {
    question: bi('A network has 5 layers and every activation is linear (identity). What can its output be, as a function of x?', 'یک شبکه ۵ لایه دارد و همه توابع فعال‌سازی‌اش خطی (همانی) هستند. خروجی‌اش بر حسب x چه شکلی می‌تواند داشته باشد؟'),
    choices: [
      { id: 'a', label: bi('Only an affine function: a straight line (or flat plane)', 'فقط یک تابع آفین: یک خط راست (یا صفحه صاف)') },
      { id: 'b', label: bi('Any curve, if it has enough neurons', 'هر منحنی‌ای، اگر نورون کافی داشته باشد') },
      { id: 'c', label: bi('Only a constant', 'فقط یک مقدار ثابت') },
    ],
    answer: 'a',
    explain: bi(
      'Each layer is affine, and an affine map of an affine map is affine. Without a nonlinear activation, depth and width cannot create a bend.',
      'هر لایه آفین است و نگاشت آفینِ یک نگاشت آفین باز آفین است. بدون تابع فعال‌سازی غیرخطی، عمق و پهنا نمی‌توانند خمی بسازند.',
    ),
  },
};
