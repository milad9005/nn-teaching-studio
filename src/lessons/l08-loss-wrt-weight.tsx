// Lesson 08: the derivative of the loss with respect to one weight.
// Shared lab store (preset "single-weight": x = 2, y = 10, w = 3, b frozen at 0), so the two
// plots, the equations, the finite-difference check and the derivative graph always agree.

import { useMemo } from 'react';
import type { LessonDef } from './types';
import { bi, useT } from '../i18n';
import { useLab, useLabDerived } from '../state/lab';
import { useLessonState } from '../state/lessonState';
import { Plot2D, type PlotLayer } from '../components/Plot2D';
import { Tex } from '../components/Tex';
import { Slider, Stat } from '../components/controls';
import { Advanced, Callout, Cycle, P, Predict, Reveal } from '../components/teaching';
import { clamp, fmt, setParam, sampleResult, tex, texP } from '../engine';
import { W_REF, finiteDiffs, lossAtW, outAtW, singleWeight, wDomain } from './math/l08';

const ID = 'l08';

interface S {
  /** Show the finite-difference comparison. */
  fd: boolean;
  /** h = 10^(-e) */
  e: number;
}
const INIT: S = { fd: true, e: 1 };
const useS = () => useLessonState<S>(ID, INIT);

/** Every number of the lesson, from the lab store and the engine. */
function useOne() {
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const loss = useLab((s) => s.loss);
  const { one } = useLabDerived();
  const x = data.X[0][0];
  const y = data.Y[0][0];
  const w = net.layers[0].W[0][0];
  const b = net.layers[0].b[0];
  const closed = singleWeight(x, y, w, b);
  return {
    net,
    data,
    loss,
    x,
    y,
    w,
    b,
    ...closed,
    // Engine values (forward pass and backprop) — identical to the closed form, shown side by side.
    yhat: one.trace.a[1][0],
    L: one.loss,
    dLdyhat: one.dOut[0],
    g: one.back.grads.W[0][0][0],
    fL: (v: number) => lossAtW(net, data, loss, v),
    fOut: (v: number) => outAtW(net, x, v),
  };
}

const round2 = (v: number) => Math.round(v * 100) / 100;

function Explain() {
  const { tr } = useT();
  return (
    <>
      <Cycle active={2} />
      <Reveal at={1}>
        <h3>{tr(bi('1. A network with one knob', '۱. شبکه‌ای با یک پیچ تنظیم'))}</h3>
        <P>
          {bi(
            'The smallest possible network: one input $\\cx{x}$, one weight $\\cw{w}$, and a bias frozen at 0. Its prediction is $\\cyhat{\\hat y} = \\cw{w}\\,\\cx{x}$ and the loss compares it with the target: $\\cL{L} = (\\cyhat{\\hat y} - \\cy{y})^2$.',
            'کوچک‌ترین شبکه ممکن: یک ورودی $\\cx{x}$، یک وزن $\\cw{w}$ و یک بایاس (bias) که روی ۰ ثابت شده. پیش‌بینی‌اش $\\cyhat{\\hat y} = \\cw{w}\\,\\cx{x}$ است و خطا (loss) آن را با هدف مقایسه می‌کند: $\\cL{L} = (\\cyhat{\\hat y} - \\cy{y})^2$.',
          )}
        </P>
        <P>
          {bi(
            'With $\\cx{x} = 2$, $\\cy{y} = 10$ and the starting weight $\\cw{w} = 3$: the network predicts $6$ and the loss is $(6-10)^2 = 16$. Training has exactly one thing it may change: $\\cw{w}$.',
            'با $\\cx{x} = 2$ و $\\cy{y} = 10$ و وزن شروع $\\cw{w} = 3$: شبکه $6$ را پیش‌بینی می‌کند و خطا $(6-10)^2 = 16$ است. آموزش فقط یک چیز را اجازه دارد عوض کند: $\\cw{w}$.',
          )}
        </P>
      </Reveal>
      <Reveal at={2}>
        <h3>{tr(bi('2. Two different questions', '۲. دو سؤال متفاوت'))}</h3>
        <P>
          {bi(
            'Moving $\\cw{w}$ changes two things, shown side by side. Left: the output $\\cyhat{\\hat y}(w)$, a straight line. Right: the loss $\\cL{L}(w)$, a bowl (parabola).',
            'جابه‌جا کردن $\\cw{w}$ دو چیز را عوض می‌کند که کنار هم نشان داده شده‌اند. چپ: خروجی $\\cyhat{\\hat y}(w)$ که یک خط راست است. راست: خطا $\\cL{L}(w)$ که یک کاسه (سهمی) است.',
          )}
        </P>
        <P>
          {bi(
            'Each has its own slope. $\\frac{d\\hat y}{dw}$ answers "how fast does the output move?". $\\cg{\\frac{dL}{dw}}$ answers "how fast does the error move?". Only the second one tells training what to do.',
            'هر کدام شیب خودش را دارد. $\\frac{d\\hat y}{dw}$ جواب می‌دهد «خروجی با چه سرعتی تغییر می‌کند؟». $\\cg{\\frac{dL}{dw}}$ جواب می‌دهد «خطا با چه سرعتی تغییر می‌کند؟». فقط دومی به آموزش می‌گوید چه کار کند.',
          )}
        </P>
      </Reveal>
      <Reveal at={3}>
        <Predict
          id={ID + '-dir'}
          question={bi(
            'At $w = 3$ the prediction (6) is below the target (10). If $w$ increases a little, does the loss go up or down?',
            'در $w = 3$ پیش‌بینی (۶) از هدف (۱۰) کمتر است. اگر $w$ کمی زیاد شود، خطا بالا می‌رود یا پایین؟',
          )}
          choices={[
            { id: 'up', label: bi('Up', 'بالا') },
            { id: 'down', label: bi('Down', 'پایین') },
            { id: 'same', label: bi('It stays the same', 'تغییری نمی‌کند') },
          ]}
          answer="down"
          explain={bi(
            'A bigger $w$ pushes $\\hat y$ from 6 toward 10, so the error shrinks: $dL/dw = -16 < 0$. Drag the point on the loss plot to the right and watch $L$ fall until $w = 5$.',
            '$w$ بزرگ‌تر، $\\hat y$ را از ۶ به سمت ۱۰ می‌برد، پس خطا کم می‌شود: $dL/dw = -16 < 0$. نقطه روی نمودار خطا را به راست بکش و ببین $L$ تا $w = 5$ پایین می‌آید.',
          )}
        />
      </Reveal>
      <Reveal at={4}>
        <h3>{tr(bi('3. Why training needs dL/dw', '۳. چرا آموزش به dL/dw نیاز دارد'))}</h3>
        <P>
          {bi(
            'The output slope is $\\frac{d\\hat y}{dw} = \\cx{x} = 2$ at every $w$: raising $w$ always raises $\\hat y$. It says nothing about whether $\\hat y$ is already too big or still too small, so it cannot tell us which way to move.',
            'شیب خروجی در هر $w$ برابر $\\frac{d\\hat y}{dw} = \\cx{x} = 2$ است: زیاد کردن $w$ همیشه $\\hat y$ را زیاد می‌کند. اما نمی‌گوید $\\hat y$ از قبل زیادی بزرگ است یا هنوز کوچک، پس نمی‌تواند بگوید به کدام طرف برویم.',
          )}
        </P>
        <P>
          {bi(
            'The loss slope adds the missing piece: the error $\\hat y - y$. A change in $w$ first moves $\\hat y$ (rate $x$), and that move changes $L$ (rate $2(\\hat y - y)$):',
            'شیب خطا تکه گمشده را اضافه می‌کند: خطای $\\hat y - y$. تغییر $w$ اول $\\hat y$ را جابه‌جا می‌کند (با نرخ $x$) و این جابه‌جایی $L$ را تغییر می‌دهد (با نرخ $2(\\hat y - y)$):',
          )}
        </P>
        <div className="formula-hero">
          <Tex display>{'\\cg{\\frac{dL}{dw}} = \\underbrace{2(\\cyhat{\\hat y}-\\cy{y})}_{dL/d\\hat y}\\cdot\\underbrace{\\cx{x}}_{d\\hat y/dw}'}</Tex>
        </div>
        <Callout>
          {tr(
            bi(
              'Below the target, ŷ − y < 0 and dL/dw < 0: increase w. Above the target, dL/dw > 0: decrease w. At the target, dL/dw = 0: stop. The sign of dL/dw is the direction to avoid.',
              'زیر هدف، ŷ − y < 0 و dL/dw < 0 است: w را زیاد کن. بالای هدف، dL/dw > 0 است: w را کم کن. روی هدف، dL/dw = 0: بایست. علامت dL/dw جهتی است که نباید رفت.',
            ),
          )}
        </Callout>
      </Reveal>
      <Reveal at={5}>
        <h3>{tr(bi('4. Check it with tiny nudges', '۴. بررسی با تکان‌های کوچک'))}</h3>
        <P>
          {bi(
            'We can measure the slope without any calculus: nudge $w$ by a small $h$, re-run the network, and divide the change in loss by $h$. The forward difference $\\frac{L(w+h)-L(w)}{h}$ and the central difference $\\frac{L(w+h)-L(w-h)}{2h}$ should both approach the exact value as $h$ shrinks.',
            'می‌توانیم شیب را بدون هیچ حسابانی اندازه بگیریم: $w$ را به اندازه یک $h$ کوچک تکان بده، شبکه را دوباره اجرا کن و تغییر خطا را بر $h$ تقسیم کن. تفاضل پیشرو $\\frac{L(w+h)-L(w)}{h}$ و تفاضل مرکزی $\\frac{L(w+h)-L(w-h)}{2h}$ هر دو باید با کوچک شدن $h$ به مقدار دقیق نزدیک شوند.',
          )}
        </P>
        <Advanced title={bi('Exact error of each approximation', 'خطای دقیق هر تقریب')}>
          <P>
            {bi(
              'Expand the loss after a nudge: $L(w+h) = (\\hat y + xh - y)^2 = L(w) + 2(\\hat y - y)x\\,h + x^2h^2$. Dividing by $h$: the forward difference is $\\frac{dL}{dw} + x^2 h$, an error of exactly $x^2h$ (here $4h$).',
              'خطا را بعد از تکان باز کن: $L(w+h) = (\\hat y + xh - y)^2 = L(w) + 2(\\hat y - y)x\\,h + x^2h^2$. با تقسیم بر $h$: تفاضل پیشرو برابر $\\frac{dL}{dw} + x^2 h$ است، یعنی خطایش دقیقاً $x^2h$ است (اینجا $4h$).',
            )}
          </P>
          <P>
            {bi(
              'In the central difference the $x^2h^2$ terms cancel, so for this parabola it is exact for every $h$; for general smooth functions its error is $O(h^2)$. For very small $h$ (around $10^{-8}$ and below) floating-point rounding takes over and both get worse.',
              'در تفاضل مرکزی جمله‌های $x^2h^2$ همدیگر را خنثی می‌کنند، پس برای این سهمی با هر $h$ دقیق است؛ برای توابع نرم در حالت کلی خطایش $O(h^2)$ است. برای $h$ خیلی کوچک (حدود $10^{-8}$ و کمتر) خطای گرد کردن اعداد اعشاری (floating-point) غالب می‌شود و هر دو بدتر می‌شوند.',
            )}
          </P>
          <P>
            {bi(
              'Edge cases: if $x < 0$ the output slope is negative and the signs flip; if $x = 0$ the weight cannot affect the output at all and $dL/dw = 0$ for every $w$. With $N$ examples, MSE averages: $\\frac{dL}{dw} = \\frac{2}{N}\\sum_n (\\hat y_n - y_n)x_n$.',
              'حالت‌های خاص: اگر $x < 0$ باشد شیب خروجی منفی است و علامت‌ها برعکس می‌شوند؛ اگر $x = 0$ باشد وزن اصلاً روی خروجی اثری ندارد و برای هر $w$ داریم $dL/dw = 0$. با $N$ نمونه، MSE میانگین می‌گیرد: $\\frac{dL}{dw} = \\frac{2}{N}\\sum_n (\\hat y_n - y_n)x_n$.',
            )}
          </P>
        </Advanced>
      </Reveal>
    </>
  );
}

function SlopeVerdict() {
  const { tr } = useT();
  const c = useOne();
  const g = c.g;
  const sign = g > 1e-9 ? 'pos' : g < -1e-9 ? 'neg' : 'zero';
  return (
    <div className="viz-grid">
      <Callout>
        <strong className="c-yhat">{tr(bi('Output slope', 'شیب خروجی'))}</strong> <Tex>{`\\frac{d\\hat y}{dw} = \\cx{x} = ${tex(c.dyhatdw)}`}</Tex>
        <div className="small">
          {Math.abs(c.x) < 1e-9
            ? tr(bi('x = 0: the weight has no effect on the output.', 'x = 0: وزن هیچ اثری روی خروجی ندارد.'))
            : tr(
                bi(
                  `The same at every w. It only says "w up → ŷ ${c.x > 0 ? 'up' : 'down'}", not whether that is good.`,
                  `در هر w همین است. فقط می‌گوید «w بالا ← ŷ ${c.x > 0 ? 'بالا' : 'پایین'}»، نه اینکه این خوب است یا بد.`,
                ),
              )}
        </div>
      </Callout>
      <Callout tone={sign === 'zero' ? 'info' : 'warn'}>
        <strong className="c-L">{tr(bi('Loss slope', 'شیب خطا'))}</strong> <Tex>{`\\cg{\\frac{dL}{dw}} = ${tex(g, 4)}`}</Tex>
        <div className="small" data-testid="l08-verdict">
          {sign === 'neg' && tr(bi('Negative: increasing w lowers the loss. Move w to the right.', 'منفی: زیاد کردن w خطا را کم می‌کند. w را به راست ببر.'))}
          {sign === 'pos' && tr(bi('Positive: increasing w raises the loss. Move w to the left.', 'مثبت: زیاد کردن w خطا را زیاد می‌کند. w را به چپ ببر.'))}
          {sign === 'zero' && tr(bi('Zero: the bottom of the bowl (or x = 0). Nothing to improve with w.', 'صفر: ته کاسه (یا x = 0). با w چیزی برای بهتر کردن نمانده.'))}
        </div>
      </Callout>
    </div>
  );
}

function Center() {
  const { tr } = useT();
  const c = useOne();
  const [s] = useS();
  const setP = useLab((st) => st.setParam);
  const dom = wDomain(c.x, c.y, c.b);
  const setW = (v: number) => setP(W_REF, round2(clamp(v, dom[0], dom[1])));
  const h = 10 ** -s.e;
  const hasStar = Number.isFinite(c.wStar);

  const outLayers = useMemo<PlotLayer[]>(
    () => [
      { kind: 'hline', at: c.y, color: 'y', label: `y = ${fmt(c.y, 2)}` },
      ...(hasStar ? [{ kind: 'vline' as const, at: c.wStar, color: 'var(--line-strong)', label: 'w*' }] : []),
      { kind: 'fn', f: c.fOut, color: 'yhat', width: 2.8 },
      { kind: 'line', p: [c.w, c.yhat], slope: c.dyhatdw, color: 'g', width: 2, dash: '7 5' },
      { kind: 'points', color: 'yhat', pts: [{ x: c.w, y: c.yhat, r: 0.1, label: `ŷ = ${fmt(c.yhat, 3)}` }] },
    ],
    [c.y, c.wStar, hasStar, c.fOut, c.w, c.yhat, c.dyhatdw],
  );

  const lossLayers = useMemo<PlotLayer[]>(() => {
    const out: PlotLayer[] = [
      ...(hasStar ? [{ kind: 'vline' as const, at: c.wStar, color: 'var(--line-strong)', label: 'w*' }] : []),
      { kind: 'fn', f: c.fL, color: 'L', width: 2.8 },
      { kind: 'line', p: [c.w, c.L], slope: c.g, color: 'g', width: 2, dash: '7 5' },
    ];
    if (s.fd)
      out.push({
        kind: 'points',
        color: 'x',
        pts: [
          { x: c.w - h, y: c.fL(c.w - h), r: 3.5, hollow: true },
          { x: c.w + h, y: c.fL(c.w + h), r: 3.5, hollow: true },
        ],
      });
    out.push({ kind: 'points', color: 'L', pts: [{ x: c.w, y: c.L, r: 0.1, label: `L = ${fmt(c.L, 3)}` }] });
    return out;
  }, [hasStar, c.wStar, c.fL, c.w, c.L, c.g, s.fd, h]);

  return (
    <div className="viz">
      <div className="legend-chips">
        <span className="chip">
          <i style={{ background: 'var(--c-yhat)' }} /> {tr(bi('output ŷ(w)', 'خروجی ŷ(w)'))}
        </span>
        <span className="chip">
          <i style={{ background: 'var(--c-L)' }} /> {tr(bi('loss L(w)', 'خطا L(w)'))}
        </span>
        <span className="chip">
          <i style={{ background: 'var(--c-g)' }} /> {tr(bi('tangent (slope)', 'مماس (شیب)'))}
        </span>
        <span className="chip">
          <i style={{ background: 'var(--c-y)' }} /> {tr(bi('target y', 'هدف y'))}
        </span>
        {s.fd && (
          <span className="chip">
            <i style={{ background: 'var(--c-x)' }} /> w ± h
          </span>
        )}
      </div>
      <div className="viz-grid">
        <div>
          <div className="small muted">{tr(bi('(1) Output as a function of w', '(۱) خروجی بر حسب w'))}</div>
          <Plot2D
            testId="l08-out-plot"
            xDomain={dom}
            yDomain="auto"
            layers={outLayers}
            height={280}
            xLabel={tr(bi('weight w', 'وزن w'))}
            yLabel="ŷ"
            handle={{ x: c.w, y: c.yhat, color: 'yhat' }}
            onDrag={(v) => setW(v)}
            ariaLabel="output as a function of the weight"
          />
        </div>
        <div>
          <div className="small muted">{tr(bi('(2) Loss as a function of w', '(۲) خطا بر حسب w'))}</div>
          <Plot2D
            testId="l08-loss-plot"
            xDomain={dom}
            yDomain="auto"
            layers={lossLayers}
            height={280}
            xLabel={tr(bi('weight w', 'وزن w'))}
            yLabel="L"
            handle={{ x: c.w, y: c.L, color: 'L' }}
            onDrag={(v) => setW(v)}
            ariaLabel="loss as a function of the weight"
          />
        </div>
      </div>
      <div className="factor-row" data-testid="l08-factors">
        <div className="factor" style={{ cursor: 'default' }}>
          <Tex>{'\\frac{dL}{d\\hat y} = 2(\\hat y - y)'}</Tex>
          <div className="fv c-g">{fmt(c.dLdyhat, 4)}</div>
          <div className="small muted">{tr(bi('how the loss reacts to ŷ', 'واکنش خطا به ŷ'))}</div>
        </div>
        <span className="factor-op">×</span>
        <div className="factor" style={{ cursor: 'default' }}>
          <Tex>{'\\frac{d\\hat y}{dw} = x'}</Tex>
          <div className="fv c-g">{fmt(c.dyhatdw, 4)}</div>
          <div className="small muted">{tr(bi('how ŷ reacts to w', 'واکنش ŷ به w'))}</div>
        </div>
        <span className="factor-op">=</span>
        <div className="factor on" style={{ cursor: 'default' }}>
          <Tex>{'\\frac{dL}{dw}'}</Tex>
          <div className="fv c-g" data-testid="l08-grad">
            {fmt(c.g, 4)}
          </div>
          <div className="small muted">{tr(bi('how the loss reacts to w', 'واکنش خطا به w'))}</div>
        </div>
      </div>
      <SlopeVerdict />
      <p className="small muted">{tr(bi('Drag on either plot to move w. Both points move together: it is the same weight.', 'روی هر کدام از نمودارها بکش تا w جابه‌جا شود. هر دو نقطه با هم حرکت می‌کنند: هر دو همان یک وزن‌اند.'))}</p>
    </div>
  );
}

function Right() {
  const { tr } = useT();
  const [s, set] = useS();
  const c = useOne();
  const setP = useLab((st) => st.setParam);
  const setInput = useLab((st) => st.setInput);
  const setTarget = useLab((st) => st.setTarget);
  const h = 10 ** -s.e;
  const d = finiteDiffs(c.fL, c.w, h, c.g);
  return (
    <div className="stack">
      <Slider label="w" value={c.w} min={-2} max={10} step={0.01} color="w" onChange={(v) => setP(W_REF, v)} testId="l08-w" />
      <Slider label="x" value={c.x} min={-3} max={4} step={0.1} color="x" onChange={(v) => setInput(0, 0, v)} testId="l08-x" />
      <Slider label="y" value={c.y} min={-5} max={20} step={0.5} color="y" onChange={(v) => setTarget(0, 0, v)} testId="l08-y" />
      <div className="small muted">
        <span className="c-b">b = {fmt(c.b, 3)}</span> · {tr(bi('frozen (not trained in this lesson)', 'ثابت (در این درس آموزش نمی‌بیند)'))}
      </div>

      <h4>{tr(bi('Forward', 'رو به جلو'))}</h4>
      <Tex display>{`\\cyhat{\\hat y} = \\cw{w}\\,\\cx{x} + \\cb{b} = ${texP(c.w)}\\cdot${texP(c.x)} + ${texP(c.b)} = ${tex(c.yhat, 4)}`}</Tex>
      <Tex display>{`\\cL{L} = (\\cyhat{\\hat y} - \\cy{y})^2 = (${tex(c.yhat, 4)} - ${texP(c.y)})^2 = ${tex(c.L, 4)}`}</Tex>

      <h4>{tr(bi('Two derivatives', 'دو مشتق'))}</h4>
      <Tex display>{`\\frac{d\\hat y}{dw} = \\cx{x} = ${tex(c.dyhatdw, 4)}`}</Tex>
      <Tex display>{`\\cg{\\frac{dL}{dw}} = 2(\\cyhat{\\hat y} - \\cy{y})\\cdot\\cx{x} = 2(${tex(c.yhat, 4)} - ${texP(c.y)})\\cdot${texP(c.x)} = \\cg{${tex(c.dLdw, 4)}}`}</Tex>
      <div className="stat-grid">
        <Stat k={tr(bi('prediction ŷ', 'پیش‌بینی ŷ'))} v={fmt(c.yhat, 4)} color="yhat" />
        <Stat k={tr(bi('loss L', 'خطا L'))} v={fmt(c.L, 4)} color="L" testId="l08-loss" />
        <Stat k="dŷ/dw" v={fmt(c.dyhatdw, 4)} />
        <Stat k={tr(bi('dL/dw (backprop)', 'dL/dw (Backprop)'))} v={fmt(c.g, 4)} color="g" testId="l08-backprop" />
      </div>

      <h4>{tr(bi('Finite-difference check', 'بررسی با تفاضل متناهی'))}</h4>
      <label className="small row" style={{ gap: 6 }}>
        <input type="checkbox" checked={s.fd} onChange={(e) => set({ fd: e.target.checked })} data-testid="l08-fd-toggle" />
        {tr(bi('Compare with finite differences', 'مقایسه با تفاضل متناهی (finite difference)'))}
      </label>
      {s.fd && (
        <>
          <Slider label={<Tex>{'h = 10^{-e},\\ e'}</Tex>} value={s.e} min={0} max={10} step={0.5} onChange={(e) => set({ e })} testId="l08-h" />
          <Tex display>{`\\text{fwd} = \\frac{L(w+h)-L(w)}{h} = \\frac{${tex(c.fL(c.w + h), 6)} - ${texP(c.L, 6)}}{${tex(h, 6)}} = \\htmlClass{c-x}{${tex(d.fwd, 6)}}`}</Tex>
          <Tex display>{`\\text{central} = \\frac{L(w+h)-L(w-h)}{2h} = \\htmlClass{c-x}{${tex(d.cen, 6)}}`}</Tex>
          <div className="stat-grid">
            <Stat k="h" v={fmt(h, 10)} />
            <Stat k={tr(bi('forward', 'پیشرو'))} v={fmt(d.fwd, 6)} color="x" testId="l08-fwd" />
            <Stat k={tr(bi('central', 'مرکزی'))} v={fmt(d.cen, 6)} color="x" testId="l08-central" />
            <Stat k={tr(bi('exact', 'دقیق'))} v={fmt(c.g, 6)} color="g" />
            <Stat k={tr(bi('forward error', 'خطای پیشرو'))} v={fmt(d.errFwd, 8)} />
            <Stat k={tr(bi('central error', 'خطای مرکزی'))} v={fmt(d.errCen, 8)} />
          </div>
          <P className="small muted">
            {bi(
              'Forward error ≈ $x^2 h$: shrink $h$ tenfold and the error shrinks tenfold, until rounding noise appears for tiny $h$.',
              'خطای پیشرو ≈ $x^2 h$: $h$ را ده برابر کوچک کن، خطا ده برابر کوچک می‌شود، تا وقتی که برای $h$ خیلی کوچک نویز گرد کردن پیدا شود.',
            )}
          </P>
        </>
      )}
    </div>
  );
}

function Bottom() {
  const { tr } = useT();
  const c = useOne();
  const [s] = useS();
  const dom = wDomain(c.x, c.y, c.b);
  // dL/dw at every w, from the engine's backprop on a copy of the network.
  const gAt = useMemo(() => (v: number) => sampleResult(setParam(c.net, W_REF, v), c.data.X[0], c.data.Y[0], c.loss).back.grads.W[0][0][0], [c.net, c.data, c.loss]);
  const rows = [1, 0.1, 0.01, 0.001, 1e-4, 1e-6, 1e-8, 1e-10];
  const hNow = 10 ** -s.e;
  return (
    <div className="viz-grid">
      <div>
        <div className="small muted">{tr(bi('Both slopes as functions of w', 'هر دو شیب بر حسب w'))}</div>
        <Plot2D
          testId="l08-slopes"
          xDomain={dom}
          yDomain="auto"
          height={220}
          xLabel="w"
          yLabel={tr(bi('slope', 'شیب'))}
          legend={[
            { label: 'dL/dw', color: 'g' },
            { label: 'dŷ/dw', color: 'yhat', dash: true },
          ]}
          layers={[
            { kind: 'hline', at: 0, color: 'var(--line-strong)' },
            { kind: 'fn', f: gAt, color: 'g', width: 2.4 },
            { kind: 'fn', f: () => c.dyhatdw, color: 'yhat', width: 2, dash: '6 4' },
            { kind: 'points', color: 'g', pts: [{ x: c.w, y: c.g, label: `dL/dw = ${fmt(c.g, 3)}` }] },
          ]}
        />
        <P className="small muted">
          {bi(
            'The output slope is flat at $x$. The loss slope crosses zero exactly at the best weight $w^* = (y-b)/x$: negative to its left, positive to its right.',
            'شیب خروجی روی $x$ صاف می‌ماند. شیب خطا دقیقاً در بهترین وزن $w^* = (y-b)/x$ از صفر رد می‌شود: سمت چپش منفی، سمت راستش مثبت.',
          )}
        </P>
      </div>
      <table className="tbl" data-testid="l08-h-table">
        <thead>
          <tr>
            <th>h</th>
            <th className="num">{tr(bi('forward', 'پیشرو'))}</th>
            <th className="num">{tr(bi('error', 'خطا'))}</th>
            <th className="num">{tr(bi('central', 'مرکزی'))}</th>
            <th className="num">{tr(bi('error', 'خطا'))}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((h) => {
            const d = finiteDiffs(c.fL, c.w, h, c.g);
            return (
              <tr key={h} className={s.fd && Math.abs(h - hNow) < h * 1e-6 ? 'hl' : ''}>
                <td className="num">{fmt(h, 10)}</td>
                <td className="num">{fmt(d.fwd, 6)}</td>
                <td className="num">{fmt(d.errFwd, 8)}</td>
                <td className="num">{fmt(d.cen, 6)}</td>
                <td className="num">{fmt(d.errCen, 8)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export const lesson08: LessonDef = {
  id: ID,
  title: bi('Derivative of Loss with Respect to Weight', 'مشتق خطا نسبت به وزن'),
  short: bi('dL/dw', 'dL/dw'),
  objectives: [
    bi('Tell apart the output curve ŷ(w) and the loss curve L(w)', 'منحنی خروجی ŷ(w) و منحنی خطا L(w) را از هم تشخیص بدهی'),
    bi('Explain why dŷ/dw alone cannot guide training', 'توضیح بدهی چرا dŷ/dw به‌تنهایی نمی‌تواند آموزش را هدایت کند'),
    bi('Compute dL/dw = 2(ŷ − y)·x and read its sign', 'dL/dw = 2(ŷ − y)·x را حساب کنی و علامتش را بخوانی'),
    bi('Verify the derivative with forward and central differences', 'مشتق را با تفاضل پیشرو و مرکزی بررسی کنی'),
  ],
  presetId: 'single-weight',
  revealSteps: 5,
  Explain,
  Center,
  Right,
  Bottom,
  checkpoint: {
    question: bi('The output slope dŷ/dw is +2 both at w = 3 and at w = 7. What does dL/dw tell us that dŷ/dw does not?', 'شیب خروجی dŷ/dw هم در w = 3 و هم در w = 7 برابر ۲+ است. dL/dw چه چیزی می‌گوید که dŷ/dw نمی‌گوید؟'),
    choices: [
      { id: 'a', label: bi('How large the input x is', 'ورودی x چقدر بزرگ است') },
      { id: 'b', label: bi('Whether ŷ is below or above the target, and so which way w should move', 'اینکه ŷ زیر هدف است یا بالای آن، و در نتیجه w باید به کدام طرف برود') },
      { id: 'c', label: bi('Nothing: they are always equal', 'هیچ: همیشه با هم برابرند') },
    ],
    answer: 'b',
    explain: bi(
      'dL/dw = 2(ŷ − y)·x carries the error ŷ − y. At w = 3 it is −16 (move right); at w = 7 it is +16 (move left). Drag w to both values to see the sign flip.',
      'dL/dw = 2(ŷ − y)·x خطای ŷ − y را در خود دارد. در w = 3 برابر ۱۶− است (به راست برو)؛ در w = 7 برابر ۱۶+ است (به چپ برو). w را به هر دو مقدار ببر و برعکس شدن علامت را ببین.',
    ),
  },
};
