// Lesson 07: Derivatives from zero. Slope of a line -> secant on a curve -> tangent as h -> 0.

import { useMemo } from 'react';
import type { LessonDef } from './types';
import { bi, useT } from '../i18n';
import { useLessonState } from '../state/lessonState';
import { Plot2D, type PlotLayer } from '../components/Plot2D';
import { Tex } from '../components/Tex';
import { Segmented, Slider, Stat } from '../components/controls';
import { Advanced, Callout, Cycle, P, Predict, Reveal } from '../components/teaching';
import { fmt, tex, texP } from '../engine';

const ID = 'l07';

// A loss-shaped curve: L(w) = 0.5 (w - 3)^2 + 1, so L'(w) = w - 3.
const Lc = (w: number) => 0.5 * (w - 3) ** 2 + 1;
const dLc = (w: number) => w - 3;

interface S {
  stage: 'line' | 'curve';
  m: number;
  c: number;
  a: number;
  /** h = 10^(-e) */
  e: number;
  zoom: boolean;
}
const INIT: S = { stage: 'line', m: 0.5, c: 1, a: 5, e: 0, zoom: false };
const useS = () => useLessonState<S>(ID, INIT);

function f(s: S) {
  return s.stage === 'line' ? (x: number) => s.m * x + s.c : Lc;
}
function df(s: S, x: number) {
  return s.stage === 'line' ? s.m : dLc(x);
}

function Explain() {
  const { tr } = useT();
  const [s, set] = useS();
  return (
    <>
      <Cycle active={s.stage === 'line' ? 0 : 2} />
      <Reveal at={1}>
        <h3>{tr(bi('1. Slope of a straight line', '۱. شیب یک خط راست'))}</h3>
        <P>
          {bi(
            'A straight line goes up by the same amount for every step to the right. That fixed ratio is its slope: $\\text{slope} = \\dfrac{\\text{rise}}{\\text{run}}$.',
            'یک خط راست به ازای هر قدم به راست، همیشه به یک اندازه بالا می‌رود. این نسبت ثابت، شیب (slope) آن است: $\\text{slope} = \\dfrac{\\text{rise}}{\\text{run}}$.',
          )}
        </P>
        <P>
          {bi(
            'Positive slope: going right means going up. Negative slope: going right means going down. Zero slope: flat.',
            'شیب مثبت: به راست بروی، بالا می‌روی. شیب منفی: به راست بروی، پایین می‌آیی. شیب صفر: صاف و افقی.',
          )}
        </P>
      </Reveal>
      <Reveal at={2}>
        <h3>{tr(bi('2. A curve has no single slope', '۲. یک منحنی یک شیب ثابت ندارد'))}</h3>
        <P>
          {bi(
            'A loss curve bends. Its steepness is different at every point, so "the slope" only makes sense at a specific point $a$.',
            'منحنی خطا خم می‌شود. شیبش در هر نقطه فرق می‌کند، پس «شیب» فقط در یک نقطه مشخص مثل $a$ معنی دارد.',
          )}
        </P>
        <P>
          {bi(
            'Idea: pick a second point a small step $h$ away and draw the line through both (a secant). Its slope is $\\dfrac{f(a+h)-f(a)}{h}$.',
            'ایده: یک نقطه دوم به فاصله کوچک $h$ انتخاب کن و خطی از هر دو نقطه بکش (خط قاطع یا secant). شیبش می‌شود $\\dfrac{f(a+h)-f(a)}{h}$.',
          )}
        </P>
        {s.stage === 'line' && (
          <button className="btn sm" onClick={() => set({ stage: 'curve', a: 5, e: 0 })} data-testid="to-curve">
            {tr(bi('Switch to the curve →', 'برو سراغ منحنی ←'))}
          </button>
        )}
      </Reveal>
      <Reveal at={3}>
        <Predict
          id={ID + '-shrink'}
          question={bi('As $h$ gets smaller, what happens to the blue secant line?', 'وقتی $h$ کوچک‌تر می‌شود، خط قاطع آبی چه می‌شود؟')}
          choices={[
            { id: 'a', label: bi('It flips direction', 'جهتش برعکس می‌شود') },
            { id: 'b', label: bi('It settles onto the tangent line', 'روی خط مماس می‌نشیند') },
            { id: 'c', label: bi('It becomes vertical', 'عمودی می‌شود') },
          ]}
          answer="b"
          explain={bi(
            'Drag the h slider to the right. The secant slope approaches one number: the derivative $f\'(a)$.',
            'اسلایدر h را به راست بکش. شیب خط قاطع به یک عدد نزدیک می‌شود: مشتق $f\'(a)$.',
          )}
        />
      </Reveal>
      <Reveal at={4}>
        <h3>{tr(bi('3. The derivative', '۳. مشتق'))}</h3>
        <Tex display>{"f'(a) = \\lim_{h\\to 0}\\frac{f(a+h)-f(a)}{h}"}</Tex>
        <P>
          {bi(
            'The derivative is the instantaneous rate of change: how fast $f$ changes per unit of input, right at $a$. Positive: uphill to the right. Negative: downhill to the right. Zero: a flat spot, like the bottom of the loss curve.',
            'مشتق نرخ تغییر لحظه‌ای است: اینکه $f$ دقیقاً در نقطه $a$ به ازای هر واحد ورودی چقدر تغییر می‌کند. مثبت: به راست سربالایی. منفی: به راست سرازیری. صفر: یک نقطه صاف، مثل ته منحنی خطا.',
          )}
        </P>
        <Callout>
          {tr(
            bi(
              'Turn on "zoom with h": up close every smooth curve looks like a straight line. That is why a derivative is a local approximation: f(a + Δ) ≈ f(a) + f′(a)·Δ only for small Δ.',
              '«زوم همراه h» را روشن کن: از نزدیک هر منحنی نرم شبیه خط راست است. برای همین مشتق یک تقریب محلی است: f(a + Δ) ≈ f(a) + f′(a)·Δ فقط برای Δهای کوچک درست است.',
            ),
          )}
        </Callout>
      </Reveal>
      <Reveal at={5}>
        <Advanced>
          <P>
            {bi(
              'One-sided limits: $h\\to0^+$ uses steps to the right only (the right-hand derivative). The derivative exists when the left and right limits agree. For ReLU at $z=0$ they do not (0 from the left, 1 from the right), so $\\relu\'(0)$ is undefined; training code picks a convention.',
              'حدهای یک‌طرفه: $h\\to0^+$ فقط قدم‌های به راست را استفاده می‌کند (مشتق راست). مشتق وقتی وجود دارد که حد چپ و راست برابر باشند. برای ReLU در $z=0$ برابر نیستند (از چپ ۰، از راست ۱)، پس $\\relu\'(0)$ تعریف نشده است؛ کد آموزش یک قرارداد انتخاب می‌کند.',
            )}
          </P>
          <P>
            {bi(
              'For this curve, $\\frac{f(a+h)-f(a)}{h} = (a-3) + \\tfrac{h}{2}$ exactly, so the error is $h/2$. The central difference $\\frac{f(a+h)-f(a-h)}{2h}$ is exact here and, in general, has error $O(h^2)$. It is what gradient checks use.',
              'برای این منحنی دقیقاً $\\frac{f(a+h)-f(a)}{h} = (a-3) + \\tfrac{h}{2}$ است، پس خطا $h/2$ است. تفاضل مرکزی $\\frac{f(a+h)-f(a-h)}{2h}$ اینجا دقیق است و در حالت کلی خطای $O(h^2)$ دارد. همان چیزی است که در بررسی گرادیان استفاده می‌شود.',
            )}
          </P>
        </Advanced>
      </Reveal>
    </>
  );
}

function usePlot(s: S) {
  const fn = f(s);
  const h = 10 ** -s.e;
  const fa = fn(s.a);
  const fah = fn(s.a + h);
  const secant = (fah - fa) / h;
  const d = df(s, s.a);
  return { fn, h, fa, fah, secant, d };
}

function Center() {
  const { tr } = useT();
  const [s, set] = useS();
  const { fn, h, fa, fah, secant, d } = usePlot(s);
  const span = s.zoom && s.stage === 'curve' ? Math.max(h * 3, 1e-3) : 4;
  const xd: [number, number] = s.stage === 'curve' ? [s.a - span, s.a + span] : [-1, 7];
  const layers = useMemo<PlotLayer[]>(() => {
    const out: PlotLayer[] = [{ kind: 'fn', f: fn, color: 'L', width: 2.8 }];
    if (s.stage === 'line') {
      const x0 = s.a;
      out.push(
        { kind: 'segment', from: [x0, fn(x0)], to: [x0 + 1, fn(x0)], color: 'x', width: 2, label: 'run = 1' },
        { kind: 'segment', from: [x0 + 1, fn(x0)], to: [x0 + 1, fn(x0 + 1)], color: 'g', width: 2, label: `rise = ${fmt(s.m, 2)}` },
        { kind: 'points', color: 'L', pts: [{ x: x0, y: fn(x0) }, { x: x0 + 1, y: fn(x0 + 1), r: 4 }] },
      );
    } else {
      out.push(
        { kind: 'line', p: [s.a, fa], slope: d, color: 'g', width: 2, dash: '7 5' },
        { kind: 'line', p: [s.a, fa], slope: secant, color: 'x', width: 2 },
        { kind: 'segment', from: [s.a, fa], to: [s.a + h, fa], color: 'x', dash: '3 3', width: 1.2 },
        { kind: 'segment', from: [s.a + h, fa], to: [s.a + h, fah], color: 'x', dash: '3 3', width: 1.2 },
        { kind: 'points', color: 'x', pts: [{ x: s.a + h, y: fah, r: 5, label: 'f(a+h)' }] },
      );
    }
    return out;
  }, [fn, s, fa, fah, h, d, secant]);
  const sign = d > 1e-9 ? 'pos' : d < -1e-9 ? 'neg' : 'zero';
  return (
    <div className="viz">
      <div className="row">
        <Segmented
          label="stage"
          value={s.stage}
          onChange={(stage) => set({ stage })}
          options={[
            { value: 'line', label: tr(bi('Straight line', 'خط راست')) },
            { value: 'curve', label: tr(bi('Loss curve', 'منحنی خطا')) },
          ]}
        />
        {s.stage === 'curve' && (
          <label className="small row" style={{ gap: 4 }}>
            <input type="checkbox" checked={s.zoom} onChange={(e) => set({ zoom: e.target.checked })} data-testid="zoom" />
            {tr(bi('Zoom with h', 'زوم همراه h'))}
          </label>
        )}
        <div className="legend-chips">
          <span className="chip">
            <i style={{ background: 'var(--c-L)' }} /> {s.stage === 'line' ? 'f(x)' : 'L(w)'}
          </span>
          {s.stage === 'curve' && (
            <>
              <span className="chip">
                <i style={{ background: 'var(--c-x)' }} /> {tr(bi('secant (slope over h)', 'خط قاطع (شیب روی h)'))}
              </span>
              <span className="chip">
                <i style={{ background: 'var(--c-g)' }} /> {tr(bi('tangent = derivative', 'مماس = مشتق'))}
              </span>
            </>
          )}
        </div>
      </div>
      <Plot2D
        testId="deriv-plot"
        xDomain={xd}
        yDomain={s.stage === 'curve' && s.zoom ? 'auto' : s.stage === 'curve' ? [0, 9] : 'auto'}
        layers={layers}
        height={360}
        xLabel={s.stage === 'line' ? 'x' : tr(bi('weight w', 'وزن w'))}
        yLabel={s.stage === 'line' ? 'f(x)' : tr(bi('loss L(w)', 'خطا L(w)'))}
        handle={{ x: s.a, y: fa, color: 'L' }}
        onDrag={(x) => set({ a: Math.round(Math.max(xd[0], Math.min(xd[1], x)) * 100) / 100 })}
        ariaLabel="derivative plot"
      />
      {s.stage === 'curve' && (
        <div className={`callout ${sign === 'zero' ? '' : 'warn'}`} data-testid="deriv-sign">
          {sign === 'pos' &&
            tr(bi(`f′(a) = ${fmt(d, 3)} > 0: moving right goes uphill. To lower the loss, move left.`, `f′(a) = ${fmt(d, 3)} > 0: به راست بروی بالا می‌روی. برای کم کردن خطا باید به چپ بروی.`))}
          {sign === 'neg' &&
            tr(bi(`f′(a) = ${fmt(d, 3)} < 0: moving right goes downhill. To lower the loss, move right.`, `f′(a) = ${fmt(d, 3)} < 0: به راست بروی پایین می‌آیی. برای کم کردن خطا به راست برو.`))}
          {sign === 'zero' && tr(bi('f′(a) = 0: a flat spot. Here it is the minimum of the loss.', 'f′(a) = 0: نقطه صاف. اینجا کمینه خطاست.'))}
        </div>
      )}
      <p className="small muted">{tr(bi('Drag on the plot to move the point.', 'روی نمودار بکش تا نقطه جابه‌جا شود.'))}</p>
    </div>
  );
}

function Right() {
  const { tr } = useT();
  const [s, set] = useS();
  const { h, fa, fah, secant, d } = usePlot(s);
  return (
    <div className="stack">
      {s.stage === 'line' ? (
        <>
          <Slider label="m (slope)" value={s.m} min={-2} max={2} step={0.05} color="g" onChange={(m) => set({ m })} />
          <Slider label="c (intercept)" value={s.c} min={-2} max={4} step={0.05} color="b" onChange={(c) => set({ c })} />
          <Slider label="a" value={s.a} min={-1} max={6} step={0.05} color="x" onChange={(a) => set({ a })} />
          <Tex display>{`f(x) = ${tex(s.m)}\\,x + ${texP(s.c)}`}</Tex>
          <Tex display>{`\\text{slope} = \\frac{f(a+1)-f(a)}{1} = \\frac{${tex(s.m * (s.a + 1) + s.c)} - ${texP(s.m * s.a + s.c)}}{1} = \\cg{${tex(s.m)}}`}</Tex>
          <P className="small muted">{bi('Same answer for every a: a line has one slope everywhere.', 'برای هر a همین جواب: خط راست همه‌جا یک شیب دارد.')}</P>
        </>
      ) : (
        <>
          <Slider label="a" value={s.a} min={-1} max={7} step={0.01} color="x" onChange={(a) => set({ a })} testId="slider-a" />
          <Slider label={<Tex>{'h = 10^{-e},\\ e'}</Tex>} value={s.e} min={0} max={5} step={0.25} onChange={(e) => set({ e })} testId="slider-h" />
          <Tex display>{'L(w) = \\tfrac12 (w-3)^2 + 1'}</Tex>
          <Tex display>{`\\frac{L(a+h)-L(a)}{h} = \\frac{${tex(fah, 6)} - ${texP(fa, 6)}}{${tex(h, 6)}} = \\htmlClass{c-x}{${tex(secant, 6)}}`}</Tex>
          <Tex display>{`L'(a) = a - 3 = \\cg{${tex(d, 6)}}`}</Tex>
          <div className="stat-grid">
            <Stat k="h" v={fmt(h, 6)} />
            <Stat k={tr(bi('secant slope', 'شیب قاطع'))} v={fmt(secant, 6)} color="x" testId="secant" />
            <Stat k="L′(a)" v={fmt(d, 6)} color="g" testId="exact-deriv" />
            <Stat k={tr(bi('gap', 'فاصله'))} v={fmt(secant - d, 6)} />
          </div>
          <P className="small muted">{bi('For this curve the gap is exactly h/2: halve h, halve the error.', 'برای این منحنی فاصله دقیقاً h/2 است: h را نصف کنی، خطا نصف می‌شود.')}</P>
        </>
      )}
    </div>
  );
}

function Bottom() {
  const { tr } = useT();
  const [s] = useS();
  if (s.stage === 'line') {
    return <P className="muted">{bi('Switch to the loss curve to see the derivative as a function.', 'برای دیدن مشتق به شکل یک تابع، به منحنی خطا برو.')}</P>;
  }
  const rows = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001];
  return (
    <div className="viz-grid">
      <Plot2D
        xDomain={[-1, 7]}
        yDomain={[-4, 4]}
        height={200}
        xLabel="w"
        yLabel="L′(w)"
        layers={[
          { kind: 'hline', at: 0, color: 'var(--line-strong)' },
          { kind: 'fn', f: dLc, color: 'g', width: 2.4 },
          { kind: 'points', color: 'g', pts: [{ x: s.a, y: dLc(s.a), label: `L′(${fmt(s.a, 2)}) = ${fmt(dLc(s.a), 2)}` }] },
        ]}
      />
      <table className="tbl" data-testid="h-table">
        <thead>
          <tr>
            <th>h</th>
            <th className="num">{tr(bi('secant slope', 'شیب قاطع'))}</th>
            <th className="num">{tr(bi('gap to L′(a)', 'فاصله تا L′(a)'))}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((h) => {
            const sl = (Lc(s.a + h) - Lc(s.a)) / h;
            return (
              <tr key={h} className={Math.abs(h - 10 ** -s.e) < h * 1e-6 ? 'hl' : ''}>
                <td className="num">{fmt(h, 6)}</td>
                <td className="num">{fmt(sl, 6)}</td>
                <td className="num">{fmt(sl - dLc(s.a), 6)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export const lesson07: LessonDef = {
  id: ID,
  title: bi('Derivatives from Zero', 'مشتق از صفر'),
  short: bi('Derivatives', 'مشتق'),
  objectives: [
    bi('Read the slope of a line as rise over run', 'شیب یک خط را به شکل «بالا رفتن تقسیم بر جلو رفتن» بخوانی'),
    bi('See a tangent line emerge from secants as h → 0', 'ببینی خط مماس چطور از خط‌های قاطع وقتی h → 0 به دست می‌آید'),
    bi('Interpret positive, negative and zero derivatives', 'مشتق مثبت، منفی و صفر را تفسیر کنی'),
    bi('Explain why a derivative is only a local approximation', 'توضیح بدهی چرا مشتق فقط یک تقریب محلی است'),
  ],
  revealSteps: 5,
  Explain,
  Center,
  Right,
  Bottom,
  checkpoint: {
    question: bi('At the bottom of a loss curve the derivative is…', 'در ته منحنی خطا، مشتق…'),
    choices: [
      { id: 'a', label: bi('as large as possible', 'بیشترین مقدار ممکن است') },
      { id: 'b', label: bi('zero: the tangent is flat', 'صفر است: خط مماس افقی است') },
      { id: 'c', label: bi('undefined', 'تعریف نشده است') },
    ],
    answer: 'b',
    explain: bi('Drag the point to w = 3. The tangent becomes horizontal and L′(3) = 0.', 'نقطه را به w = 3 ببر. خط مماس افقی می‌شود و L′(3) = 0.'),
  },
};
