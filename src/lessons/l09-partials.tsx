// Lesson 09: two weights, partial derivatives and the gradient vector.
// Shared lab store (preset "two-weight": ŷ = w1·x1 + w2·x2, three examples, bias frozen at 0,
// minimum at (2, 1)). The 3-D surface, the contour map and the 1-D slices are all sampled from
// the engine's datasetLoss; arrows and equations use the engine's gradient.

import { useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import type { LessonDef } from './types';
import { bi, useT } from '../i18n';
import { useLab, useLabDerived } from '../state/lab';
import { useLessonState } from '../state/lessonState';
import { Plot2D, type PlotLayer } from '../components/Plot2D';
import { PlotlyChart, cssVar, useThemeKey } from '../components/PlotlyChart';
import { Tex } from '../components/Tex';
import { Slider, Stat } from '../components/controls';
import { Advanced, Callout, Cycle, P, Predict, Reveal } from '../components/teaching';
import { clamp, datasetLoss, fmt, linspace, predict, setParam, tex, texP } from '../engine';
import { B_REF, W1_REF, W2_REF, contourLevels, contourSegments, lossAt, lossGrid, withWeights } from './math/l09';

const ID = 'l09';

interface S {
  comp: boolean;
  grad: boolean;
  neg: boolean;
  /** Treat the (normally frozen) bias as a third parameter. */
  third: boolean;
}
const INIT: S = { comp: true, grad: true, neg: true, third: false };
const useS = () => useLessonState<S>(ID, INIT);

const W1D: [number, number] = [-1, 5];
const W2D: [number, number] = [-2, 4];
const BD: [number, number] = [-3, 3];
/** Arrows are drawn at this fraction of the gradient's length so they fit on the map. */
const ARROW = 0.25;
const r2 = (v: number) => Math.round(v * 100) / 100;

function useTwo() {
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const loss = useLab((s) => s.loss);
  const { full } = useLabDerived();
  const w1 = net.layers[0].W[0][0];
  const w2 = net.layers[0].W[0][1];
  const b = net.layers[0].b[0];
  const yhat = data.X.map((x) => predict(net, x)[0]);
  return {
    net,
    data,
    loss,
    w1,
    w2,
    b,
    yhat,
    res: yhat.map((v, n) => v - data.Y[n][0]),
    L: full.loss,
    g1: full.grads.W[0][0][0],
    g2: full.grads.W[0][0][1],
    gb: full.grads.b[0][0],
  };
}

/** Surface grid for the 3-D plot and a wider grid for contour lines. Recomputed only when b or the data change. */
function useGrids() {
  const c = useTwo();
  return useMemo(() => {
    const base = setParam(c.net, B_REF, c.b);
    const w1s = linspace(W1D[0], W1D[1], 41);
    const w2s = linspace(W2D[0], W2D[1], 41);
    const z = lossGrid(base, c.data, c.loss, w1s, w2s);
    const cx = linspace(-6, 10, 97);
    const cy = linspace(W2D[0], W2D[1], 37);
    const cz = lossGrid(base, c.data, c.loss, cx, cy);
    let min = { v: Infinity, w1: 0, w2: 0 };
    z.forEach((row, j) => row.forEach((v, i) => v < min.v && (min = { v, w1: w1s[i], w2: w2s[j] })));
    const max = Math.max(...z.flat());
    return { w1s, w2s, z, cx, cy, cz, min, levels: contourLevels(max, 11) };
    // The grid depends on everything except the two plotted weights.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [c.b, c.data, c.loss, c.net.layers[0].act]);
}

function useWidth() {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(420);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(220, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

function Explain() {
  const { tr } = useT();
  return (
    <>
      <Cycle active={2} />
      <Reveal at={1}>
        <h3>{tr(bi('1. Two knobs, one loss', '۱. دو پیچ، یک خطا'))}</h3>
        <P>
          {bi(
            'Now the neuron has two inputs and two weights: $\\cyhat{\\hat y} = \\cw{w_1}\\cx{x_1} + \\cw{w_2}\\cx{x_2}$ (bias frozen at 0). It must fit three examples, and the loss is their average squared error.',
            'حالا نورون دو ورودی و دو وزن دارد: $\\cyhat{\\hat y} = \\cw{w_1}\\cx{x_1} + \\cw{w_2}\\cx{x_2}$ (بایاس روی ۰ ثابت است). باید با سه نمونه جور شود و خطا میانگین مربع خطاهای آنهاست.',
          )}
        </P>
        <P>
          {bi(
            'The loss now depends on two numbers, so it is a surface over the $(w_1, w_2)$ plane instead of a curve. Its lowest point is at $(2, 1)$, where all three examples are fit exactly.',
            'خطا حالا به دو عدد وابسته است، پس به جای یک منحنی، یک رویه (surface) روی صفحه $(w_1, w_2)$ است. پایین‌ترین نقطه‌اش در $(2, 1)$ است، جایی که هر سه نمونه دقیقاً درست پیش‌بینی می‌شوند.',
          )}
        </P>
      </Reveal>
      <Reveal at={2}>
        <h3>{tr(bi('2. Partial derivative = slope of a slice', '۲. مشتق جزئی = شیب یک برش'))}</h3>
        <P>
          {bi(
            'Freeze $w_2$ and move only $w_1$: you walk along a line on the surface, and the loss along that line is an ordinary 1-D curve. Its slope is the partial derivative $\\cg{\\frac{\\partial L}{\\partial w_1}}$ — "how fast $L$ changes with $w_1$ while $w_2$ is held fixed".',
            '$w_2$ را ثابت نگه دار و فقط $w_1$ را حرکت بده: روی یک خط از رویه راه می‌روی و خطا در طول آن خط یک منحنی یک‌بعدی معمولی است. شیب آن، مشتق جزئی (partial derivative) $\\cg{\\frac{\\partial L}{\\partial w_1}}$ است — «$L$ با تغییر $w_1$ چقدر سریع تغییر می‌کند، وقتی $w_2$ ثابت است».',
          )}
        </P>
        <P>
          {bi(
            'Same for $\\cg{\\frac{\\partial L}{\\partial w_2}}$ with $w_1$ frozen. The curly $\\partial$ is only a reminder that other variables exist and are being held still. Both slices are drawn in the bottom panel.',
            'برای $\\cg{\\frac{\\partial L}{\\partial w_2}}$ هم همین‌طور، با $w_1$ ثابت. علامت $\\partial$ فقط یادآوری می‌کند که متغیرهای دیگری هم هستند و ثابت نگه داشته شده‌اند. هر دو برش در پنل پایین رسم شده‌اند.',
          )}
        </P>
      </Reveal>
      <Reveal at={3}>
        <Predict
          id={ID + '-dir'}
          question={bi(
            'At the start $(w_1, w_2) = (0.5, -1)$, if we increase only $w_2$ a little (keeping $w_1$ fixed), does the loss go up or down?',
            'در شروع $(w_1, w_2) = (0.5, -1)$، اگر فقط $w_2$ را کمی زیاد کنیم ($w_1$ ثابت بماند)، خطا بالا می‌رود یا پایین؟',
          )}
          choices={[
            { id: 'up', label: bi('Up', 'بالا') },
            { id: 'down', label: bi('Down', 'پایین') },
            { id: 'same', label: bi('No change', 'تغییری نمی‌کند') },
          ]}
          answer="down"
          explain={bi(
            'Examples 2 and 3 predict too little, and both use $x_2 = 1$, so a bigger $w_2$ helps: $\\partial L/\\partial w_2 = -11/3 < 0$. Drag the point on the w₂ slice (w₁ held fixed) in the bottom panel to the right.',
            'نمونه‌های ۲ و ۳ کمتر از هدف پیش‌بینی می‌کنند و هر دو $x_2 = 1$ دارند، پس $w_2$ بزرگ‌تر کمک می‌کند: $\\partial L/\\partial w_2 = -11/3 < 0$. نقطه را روی برش w₂ (با w₁ ثابت) در پنل پایین به سمت مقدارهای بزرگ‌تر بکش.',
          )}
        />
      </Reveal>
      <Reveal at={4}>
        <h3>{tr(bi('3. The gradient is a vector', '۳. گرادیان یک بردار است'))}</h3>
        <P>
          {bi(
            'Stack the partial derivatives into one vector, the gradient. On the contour map its components are the two colored arrows, and together they form the magenta arrow:',
            'مشتق‌های جزئی را در یک بردار کنار هم بگذار: گرادیان (gradient). روی نقشه تراز، مؤلفه‌هایش دو فلش رنگی‌اند و با هم فلش سرخابی را می‌سازند:',
          )}
        </P>
        <div className="formula-hero">
          <Tex display>{'\\cg{\\nabla L} = \\begin{pmatrix} \\partial L/\\partial w_1 \\\\ \\partial L/\\partial w_2 \\end{pmatrix}'}</Tex>
        </div>
        <P>
          {bi(
            'The gradient points in the direction where the loss rises fastest, and it is perpendicular to the contour line through the point. Its opposite, $-\\nabla L$ (gray dashed), is the direction of steepest descent: the way training will move.',
            'گرادیان به جهتی اشاره می‌کند که خطا در آن سریع‌تر از همه بالا می‌رود و بر خط تراز گذرنده از نقطه عمود است. جهت مخالفش، $-\\nabla L$ (خط‌چین خاکستری)، جهت تندترین سرازیری است: همان راهی که آموزش می‌رود.',
          )}
        </P>
      </Reveal>
      <Reveal at={5}>
        <h3>{tr(bi('4. More than two weights', '۴. بیش از دو وزن'))}</h3>
        <P>
          {bi(
            'With $n$ parameters the loss lives in $n+1$ dimensions. Nobody can draw that. What we can always draw is a slice: hold all parameters but one fixed (a curve) or all but two (a surface). Real networks have thousands to billions of parameters, so every loss picture you see is such a slice.',
            'با $n$ پارامتر، خطا در $n+1$ بُعد زندگی می‌کند. هیچ‌کس نمی‌تواند آن را رسم کند. چیزی که همیشه می‌توانیم رسم کنیم یک برش است: همه پارامترها جز یکی را ثابت نگه داریم (یک منحنی) یا همه جز دو تا (یک رویه). شبکه‌های واقعی هزاران تا میلیاردها پارامتر دارند، پس هر تصویری از خطا که می‌بینی چنین برشی است.',
          )}
        </P>
        <Callout>
          {tr(
            bi(
              'Try it: tick "use b as a 3rd parameter" on the right and move b. The 3-D surface is now only the slice at that b, and its lowest point is no longer zero. The gradient simply gains a third component ∂L/∂b.',
              'امتحان کن: در سمت راست «b را پارامتر سوم کن» را بزن و b را حرکت بده. رویه سه‌بعدی حالا فقط برش در آن b است و پایین‌ترین نقطه‌اش دیگر صفر نیست. گرادیان فقط یک مؤلفه سوم ∂L/∂b می‌گیرد.',
            ),
          )}
        </Callout>
        <Advanced title={bi('Directional derivatives and steepest ascent', 'مشتق جهتی و تندترین سربالایی')}>
          <P>
            {bi(
              'Moving a small step $\\varepsilon$ in a unit direction $u$ changes the loss by about $\\varepsilon\\,(\\nabla L\\cdot u)$. By Cauchy–Schwarz, $\\nabla L\\cdot u \\le \\lVert\\nabla L\\rVert$ with equality when $u = \\nabla L/\\lVert\\nabla L\\rVert$: the gradient is the steepest-ascent direction, and its length is that steepest slope.',
              'یک قدم کوچک $\\varepsilon$ در جهت واحد $u$، خطا را تقریباً به اندازه $\\varepsilon\\,(\\nabla L\\cdot u)$ تغییر می‌دهد. طبق نامساوی کوشی–شوارتز، $\\nabla L\\cdot u \\le \\lVert\\nabla L\\rVert$ و تساوی وقتی است که $u = \\nabla L/\\lVert\\nabla L\\rVert$: گرادیان جهت تندترین سربالایی است و طولش همان بیشترین شیب است.',
            )}
          </P>
          <P>
            {bi(
              'Along a contour line the loss is constant, so $\\nabla L\\cdot u = 0$ for the tangent direction $u$: the gradient is perpendicular to the contours. (The map keeps equal scales on both axes so you can see the right angle.)',
              'در طول یک خط تراز خطا ثابت است، پس برای جهت مماس $u$ داریم $\\nabla L\\cdot u = 0$: گرادیان بر خطوط تراز عمود است. (نقشه روی هر دو محور مقیاس یکسان دارد تا زاویه قائمه را ببینی.)',
            )}
          </P>
          <P>
            {bi(
              'For this linear model with MSE: $\\frac{\\partial L}{\\partial w_i} = \\frac{2}{N}\\sum_n (\\hat y_n - y_n)\\,x_{n,i}$. Each example pushes on $w_i$ in proportion to its own error times its own input $x_{n,i}$.',
              'برای این مدل خطی با MSE: $\\frac{\\partial L}{\\partial w_i} = \\frac{2}{N}\\sum_n (\\hat y_n - y_n)\\,x_{n,i}$. هر نمونه به نسبت خطای خودش ضرب در ورودی خودش $x_{n,i}$ روی $w_i$ فشار می‌آورد.',
            )}
          </P>
        </Advanced>
      </Reveal>
    </>
  );
}

function Surface3D() {
  const c = useTwo();
  const g = useGrids();
  const theme = useThemeKey();
  const { tr } = useT();
  const surface = useMemo<Data>(
    () =>
      ({
        type: 'surface',
        x: g.w1s,
        y: g.w2s,
        z: g.z,
        colorscale: 'YlOrRd',
        reversescale: true,
        showscale: false,
        opacity: 0.9,
        contours: { z: { show: true, usecolormap: true, project: { z: true }, width: 1 } },
        hovertemplate: 'w₁ = %{x:.2f}<br>w₂ = %{y:.2f}<br>L = %{z:.3f}<extra></extra>',
      }) as Data,
    [g],
  );
  const data = useMemo<Data[]>(() => {
    // The two 1-D slices through the current point, drawn on the surface.
    const s1 = g.w1s.map((v) => lossAt(c.net, c.data, c.loss, v, c.w2));
    const s2 = g.w2s.map((v) => lossAt(c.net, c.data, c.loss, c.w1, v));
    return [
      surface,
      { type: 'scatter3d', mode: 'lines', x: g.w1s, y: g.w1s.map(() => c.w2), z: s1, line: { color: cssVar('--c-wpos'), width: 6 }, hoverinfo: 'skip' } as Data,
      { type: 'scatter3d', mode: 'lines', x: g.w2s.map(() => c.w1), y: g.w2s, z: s2, line: { color: cssVar('--c-wneg'), width: 6 }, hoverinfo: 'skip' } as Data,
      {
        type: 'scatter3d',
        mode: 'markers',
        x: [c.w1],
        y: [c.w2],
        z: [c.L],
        marker: { size: 6, color: cssVar('--c-w'), line: { color: cssVar('--surface'), width: 2 } },
        hovertemplate: 'w₁ = %{x:.2f}<br>w₂ = %{y:.2f}<br>L = %{z:.4f}<extra></extra>',
      } as Data,
    ];
    // theme: re-resolve CSS colors when the theme changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [surface, g, c.net, c.data, c.loss, c.w1, c.w2, c.L, theme]);
  const layout = useMemo<Partial<Layout>>(() => {
    const ink = cssVar('--ink-2');
    const line = cssVar('--line');
    const ax = (t: string) => ({ title: { text: t }, color: ink, gridcolor: line, zerolinecolor: cssVar('--line-strong'), backgroundcolor: 'rgba(0,0,0,0)' });
    return {
      uirevision: 'l09',
      margin: { l: 0, r: 0, t: 0, b: 0 },
      scene: { xaxis: ax('w₁'), yaxis: ax('w₂'), zaxis: ax('L'), camera: { eye: { x: 1.55, y: -1.55, z: 0.95 } }, aspectmode: 'cube' },
    } as Partial<Layout>;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme]);
  return (
    <div>
      <div className="small muted">{tr(bi('Loss surface L(w₁, w₂) — drag to rotate, scroll to zoom', 'رویه خطا L(w₁, w₂) — برای چرخاندن بکش، برای زوم اسکرول کن'))}</div>
      <PlotlyChart data={data} layout={layout} height={360} testId="l09-surface" />
    </div>
  );
}

function ContourMap() {
  const { tr } = useT();
  const c = useTwo();
  const g = useGrids();
  const [s] = useS();
  const setNet = useLab((st) => st.setNet);
  const [ref, W] = useWidth();
  // Equal scales on both axes, so the gradient visibly meets the contours at a right angle.
  const H = clamp(W - 4, 240, 400);
  const spanX = (W2D[1] - W2D[0]) * ((W - 68) / (H - 64));
  const xd: [number, number] = [2 - spanX / 2, 2 + spanX / 2];
  const layers = useMemo<PlotLayer[]>(() => {
    const out: PlotLayer[] = [];
    for (const lv of g.levels)
      for (const [a, b] of contourSegments(g.cx, g.cy, g.cz, lv)) out.push({ kind: 'segment', from: a, to: b, color: 'L', opacity: 0.32, width: 1.2 });
    for (const [a, b] of contourSegments(g.cx, g.cy, g.cz, c.L)) out.push({ kind: 'segment', from: a, to: b, color: 'L', width: 2.2 });
    // The two slices through the point: w2 held fixed (horizontal) and w1 held fixed (vertical).
    out.push(
      { kind: 'segment', from: [W1D[0], c.w2], to: [W1D[1], c.w2], color: 'wpos', dash: '5 5', opacity: 0.6 },
      { kind: 'segment', from: [c.w1, W2D[0]], to: [c.w1, W2D[1]], color: 'wneg', dash: '5 5', opacity: 0.6 },
      { kind: 'points', color: 'L', pts: [{ x: g.min.w1, y: g.min.w2, r: 4.5, hollow: true, label: tr(bi('lowest grid point', 'پایین‌ترین نقطه شبکه')) }] },
    );
    const P0: [number, number] = [c.w1, c.w2];
    if (s.comp)
      out.push(
        { kind: 'segment', from: P0, to: [c.w1 + ARROW * c.g1, c.w2], color: 'wpos', width: 2.4, arrow: true },
        { kind: 'segment', from: P0, to: [c.w1, c.w2 + ARROW * c.g2], color: 'wneg', width: 2.4, arrow: true },
      );
    if (s.grad) out.push({ kind: 'segment', from: P0, to: [c.w1 + ARROW * c.g1, c.w2 + ARROW * c.g2], color: 'g', width: 2.8, arrow: true, label: '∇L' });
    if (s.neg) out.push({ kind: 'segment', from: P0, to: [c.w1 - ARROW * c.g1, c.w2 - ARROW * c.g2], color: 'lr', width: 2.4, dash: '6 4', arrow: true, label: '−∇L' });
    return out;
  }, [g, c.L, c.w1, c.w2, c.g1, c.g2, s.comp, s.grad, s.neg, tr]);
  return (
    <div ref={ref}>
      <div className="small muted">{tr(bi('Contour map (top view) — drag to move (w₁, w₂)', 'نقشه تراز (نمای بالا) — بکش تا (w₁, w₂) جابه‌جا شود'))}</div>
      <Plot2D
        testId="l09-contour"
        xDomain={xd}
        yDomain={W2D}
        layers={layers}
        height={H}
        xLabel="w₁"
        yLabel="w₂"
        handle={{ x: c.w1, y: c.w2, color: 'w' }}
        onDrag={(a, b) => setNet(withWeights(c.net, r2(clamp(a, W1D[0], W1D[1])), r2(clamp(b, W2D[0], W2D[1]))), true)}
        ariaLabel="contour map of the loss"
      />
    </div>
  );
}

function Center() {
  const { tr } = useT();
  const [s, set] = useS();
  const toggle = (k: 'comp' | 'grad' | 'neg', label: string, color: string) => (
    <label className="chip" style={{ cursor: 'pointer' }}>
      <input type="checkbox" checked={s[k]} onChange={(e) => set({ [k]: e.target.checked })} data-testid={`l09-${k}`} />
      <i style={{ background: `var(--c-${color})` }} /> {label}
    </label>
  );
  return (
    <div className="viz">
      <div className="legend-chips">
        {toggle('comp', tr(bi('∂L/∂w₁ (w₂ fixed), ∂L/∂w₂ (w₁ fixed)', '∂L/∂w₁ (w₂ ثابت)، ∂L/∂w₂ (w₁ ثابت)')), 'wpos')}
        {toggle('grad', tr(bi('gradient ∇L', 'گرادیان ∇L')), 'g')}
        {toggle('neg', tr(bi('negative gradient −∇L', 'منفی گرادیان −∇L')), 'lr')}
        <span className="chip">
          <i style={{ background: 'var(--c-L)' }} /> {tr(bi('contour through the point', 'خط تراز گذرنده از نقطه'))}
        </span>
      </div>
      <div className="viz-grid">
        <Surface3D />
        <ContourMap />
      </div>
      <p className="small muted">
        {tr(
          bi(
            `Arrows are drawn at ${ARROW}× the gradient's length. Teal = w₁ direction, orange = w₂ direction (the dashed lines are the two slices in the bottom panel).`,
            `فلش‌ها با ${ARROW} برابر طول گرادیان رسم شده‌اند. سبزآبی = جهت w₁، نارنجی = جهت w₂ (خط‌چین‌ها همان دو برش پنل پایین‌اند).`,
          ),
        )}
      </p>
    </div>
  );
}

function Right() {
  const { tr } = useT();
  const [s, set] = useS();
  const c = useTwo();
  const setP = useLab((st) => st.setParam);
  const N = c.data.X.length;
  const sumTerms = (i: number) => c.res.map((r, n) => `${texP(r)}\\cdot${texP(c.data.X[n][i])}`).join(' + ');
  const norm = Math.hypot(c.g1, c.g2, s.third ? c.gb : 0);
  return (
    <div className="stack">
      <Slider label="w₁" value={c.w1} min={W1D[0]} max={W1D[1]} step={0.01} color="w" onChange={(v) => setP(W1_REF, v)} testId="l09-w1" />
      <Slider label="w₂" value={c.w2} min={W2D[0]} max={W2D[1]} step={0.01} color="w" onChange={(v) => setP(W2_REF, v)} testId="l09-w2" />
      <label className="small row" style={{ gap: 6 }}>
        <input
          type="checkbox"
          checked={s.third}
          onChange={(e) => {
            set({ third: e.target.checked });
            if (!e.target.checked) setP(B_REF, 0);
          }}
          data-testid="l09-third"
        />
        {tr(bi('Use b as a 3rd parameter', 'b را پارامتر سوم کن'))}
      </label>
      {s.third && <Slider label="b" value={c.b} min={BD[0]} max={BD[1]} step={0.05} color="b" onChange={(v) => setP(B_REF, v)} testId="l09-b" />}

      <table className="tbl" data-testid="l09-data">
        <thead>
          <tr>
            <th>#</th>
            <th className="num c-x">x₁</th>
            <th className="num c-x">x₂</th>
            <th className="num c-y">y</th>
            <th className="num c-yhat">ŷ</th>
            <th className="num">ŷ − y</th>
          </tr>
        </thead>
        <tbody>
          {c.data.X.map((x, n) => (
            <tr key={n}>
              <td>{n + 1}</td>
              <td className="num">{fmt(x[0], 3)}</td>
              <td className="num">{fmt(x[1], 3)}</td>
              <td className="num">{fmt(c.data.Y[n][0], 3)}</td>
              <td className="num">{fmt(c.yhat[n], 3)}</td>
              <td className="num">{fmt(c.res[n], 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <Tex display>{`\\cL{L} = \\frac{1}{${N}}\\sum_n(\\cyhat{\\hat y_n}-\\cy{y_n})^2 = \\frac{${c.res.map((r) => `${texP(r)}^2`).join(' + ')}}{${N}} = ${tex(c.L, 4)}`}</Tex>
      <h4>{tr(bi('Partial derivatives', 'مشتق‌های جزئی'))}</h4>
      <Tex display>{`\\htmlClass{c-pos}{\\frac{\\partial L}{\\partial w_1}} = \\frac{2}{${N}}\\sum_n (\\hat y_n - y_n)\\,\\cx{x_{n,1}} = \\frac{2}{${N}}\\big(${sumTerms(0)}\\big) = ${tex(c.g1, 4)}`}</Tex>
      <Tex display>{`\\htmlClass{c-neg}{\\frac{\\partial L}{\\partial w_2}} = \\frac{2}{${N}}\\sum_n (\\hat y_n - y_n)\\,\\cx{x_{n,2}} = \\frac{2}{${N}}\\big(${sumTerms(1)}\\big) = ${tex(c.g2, 4)}`}</Tex>
      {s.third && <Tex display>{`\\frac{\\partial L}{\\partial \\cb{b}} = \\frac{2}{${N}}\\sum_n (\\hat y_n - y_n) = ${tex(c.gb, 4)}`}</Tex>}
      <Tex display>
        {s.third
          ? `\\cg{\\nabla L} = \\begin{pmatrix}${tex(c.g1, 4)}\\\\${tex(c.g2, 4)}\\\\${tex(c.gb, 4)}\\end{pmatrix},\\quad \\lVert\\nabla L\\rVert = ${tex(norm, 4)}`
          : `\\cg{\\nabla L} = \\begin{pmatrix}${tex(c.g1, 4)}\\\\${tex(c.g2, 4)}\\end{pmatrix},\\quad \\lVert\\nabla L\\rVert = ${tex(norm, 4)}`}
      </Tex>
      <div className="stat-grid">
        <Stat k={tr(bi('loss L', 'خطا L'))} v={fmt(c.L, 4)} color="L" testId="l09-loss" />
        <Stat k="∂L/∂w₁" v={<span className="c-pos">{fmt(c.g1, 4)}</span>} testId="l09-g1" />
        <Stat k="∂L/∂w₂" v={<span className="c-neg">{fmt(c.g2, 4)}</span>} testId="l09-g2" />
        {s.third && <Stat k="∂L/∂b" v={fmt(c.gb, 4)} color="b" />}
        <Stat k="‖∇L‖" v={fmt(norm, 4)} color="g" />
      </div>
      <P className="small muted">
        {bi(
          'Example 2 has $x_{2,1} = 0$, so it cannot push on $w_1$ at all; example 1 has $x_{1,2} = 0$ and cannot push on $w_2$.',
          'نمونه ۲ مقدار $x_{2,1} = 0$ دارد، پس اصلاً نمی‌تواند روی $w_1$ فشار بیاورد؛ نمونه ۱ مقدار $x_{1,2} = 0$ دارد و نمی‌تواند روی $w_2$ فشار بیاورد.',
        )}
      </P>
    </div>
  );
}

function Slice({
  title,
  domain,
  f,
  at,
  value,
  slope,
  color,
  name,
  onDrag,
  testId,
}: {
  title: string;
  domain: [number, number];
  f: (v: number) => number;
  at: number;
  value: number;
  slope: number;
  color: string;
  name: string;
  onDrag: (v: number) => void;
  testId: string;
}) {
  return (
    <div>
      <div className="small muted">{title}</div>
      <Plot2D
        testId={testId}
        xDomain={domain}
        yDomain="auto"
        height={200}
        xLabel={name}
        yLabel="L"
        layers={[
          { kind: 'fn', f, color, width: 2.6 },
          { kind: 'line', p: [at, value], slope, color: 'g', width: 2, dash: '6 4' },
          { kind: 'points', color: 'L', pts: [{ x: at, y: value, r: 0.1, label: `∂L/∂${name} = ${fmt(slope, 3)}` }] },
        ]}
        handle={{ x: at, y: value, color: 'L' }}
        onDrag={(v) => onDrag(r2(clamp(v, domain[0], domain[1])))}
        ariaLabel={title}
      />
    </div>
  );
}

function Bottom() {
  const { tr } = useT();
  const [s] = useS();
  const c = useTwo();
  const setP = useLab((st) => st.setParam);
  const lossB = (v: number) => datasetLoss(setParam(c.net, B_REF, v), c.data, c.loss);
  return (
    <div className="stack">
      <div className="viz-grid">
        <Slice
          testId="l09-slice-w1"
          title={tr(bi(`L(w₁) with w₂ held fixed at ${fmt(c.w2, 2)}`, `L(w₁) با w₂ ثابت روی ${fmt(c.w2, 2)}`))}
          domain={W1D}
          f={(v) => lossAt(c.net, c.data, c.loss, v, c.w2)}
          at={c.w1}
          value={c.L}
          slope={c.g1}
          color="wpos"
          name="w₁"
          onDrag={(v) => setP(W1_REF, v)}
        />
        <Slice
          testId="l09-slice-w2"
          title={tr(bi(`L(w₂) with w₁ held fixed at ${fmt(c.w1, 2)}`, `L(w₂) با w₁ ثابت روی ${fmt(c.w1, 2)}`))}
          domain={W2D}
          f={(v) => lossAt(c.net, c.data, c.loss, c.w1, v)}
          at={c.w2}
          value={c.L}
          slope={c.g2}
          color="wneg"
          name="w₂"
          onDrag={(v) => setP(W2_REF, v)}
        />
        {s.third && (
          <Slice
            testId="l09-slice-b"
            title={tr(bi(`L(b) with w₁ = ${fmt(c.w1, 2)} and w₂ = ${fmt(c.w2, 2)} held fixed`, `L(b) با w₁ = ${fmt(c.w1, 2)} و w₂ = ${fmt(c.w2, 2)} ثابت`))}
            domain={BD}
            f={lossB}
            at={c.b}
            value={c.L}
            slope={c.gb}
            color="b"
            name="b"
            onDrag={(v) => setP(B_REF, v)}
          />
        )}
      </div>
      <P className="small muted">
        {s.third
          ? bi(
              'Three parameters, three 1-D slices — one per partial derivative. The surface above is itself only the 2-D slice at the current $b$; the full landscape $L(w_1, w_2, b)$ is 4-dimensional and cannot be drawn.',
              'سه پارامتر، سه برش یک‌بعدی — یکی برای هر مشتق جزئی. رویه بالا خودش فقط برش دوبعدی در $b$ فعلی است؛ چشم‌انداز کامل $L(w_1, w_2, b)$ چهاربعدی است و رسم‌شدنی نیست.',
            )
          : bi(
              'Each curve is the surface cut along one dashed line of the contour map. The slope of the magenta dashed tangent is that partial derivative. Drag a point to move only that weight.',
              'هر منحنی، برش رویه در امتداد یکی از خط‌چین‌های نقشه تراز است. شیب مماس خط‌چین همان مشتق جزئی است. نقطه را بکش تا فقط همان وزن حرکت کند.',
            )}
      </P>
    </div>
  );
}

export const lesson09: LessonDef = {
  id: ID,
  title: bi('Multiple Weights and Partial Derivatives', 'چند وزن و مشتق‌های جزئی'),
  short: bi('Partial derivatives', 'مشتق جزئی'),
  objectives: [
    bi('Read a loss surface over two weights in 3-D and as a contour map', 'رویه خطا روی دو وزن را به شکل سه‌بعدی و نقشه تراز بخوانی'),
    bi('Interpret ∂L/∂w₁ as the slope of a slice with w₂ held fixed', '∂L/∂w₁ را به شکل شیب یک برش با w₂ ثابت تفسیر کنی'),
    bi('Assemble the gradient vector and see that −∇L points downhill', 'بردار گرادیان را بسازی و ببینی −∇L به سمت سرازیری است'),
    bi('Explain why landscapes with ≥ 3 parameters are only ever seen as slices', 'توضیح بدهی چرا چشم‌انداز با ۳ پارامتر یا بیشتر را فقط به شکل برش می‌بینیم'),
  ],
  presetId: 'two-weight',
  revealSteps: 5,
  Explain,
  Center,
  Right,
  Bottom,
  checkpoint: {
    question: bi('At some point ∇L = (−3, 0). Which statement is true?', 'در نقطه‌ای ∇L = (−3, 0) است. کدام جمله درست است؟'),
    choices: [
      { id: 'a', label: bi('We are at the minimum, because one component is 0', 'در کمینه هستیم، چون یک مؤلفه صفر است') },
      { id: 'b', label: bi('Increasing w₁ (w₂ fixed) lowers the loss; small changes of w₂ alone barely change it', 'زیاد کردن w₁ (با w₂ ثابت) خطا را کم می‌کند؛ تغییر کوچک w₂ به‌تنهایی تقریباً اثری ندارد') },
      { id: 'c', label: bi('Increasing w₂ lowers the loss fastest', 'زیاد کردن w₂ خطا را از همه سریع‌تر کم می‌کند') },
    ],
    answer: 'b',
    explain: bi(
      'Each component is the slope of its own slice. ∂L/∂w₁ = −3 < 0: the w₁ slice goes downhill to the right. ∂L/∂w₂ = 0: the w₂ slice is flat at this point. The minimum needs every component to be 0.',
      'هر مؤلفه شیب برش خودش است. ∂L/∂w₁ = −3 < 0: برش w₁ به راست سرازیر است. ∂L/∂w₂ = 0: برش w₂ در این نقطه صاف است. کمینه وقتی است که همه مؤلفه‌ها صفر باشند.',
    ),
  },
};
