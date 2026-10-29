// Lesson 2 · The machine's knobs: a neuron. ŷ = w·x + b; w is the steepness of the line, b shifts it.
// Script: docs/storyboard.md, lesson 2.
//
// One scene: the machine from lesson 1 (now with knobs) on the left, its graph on the right, and a
// strip underneath for the slope card, the formula and, at the end, the sliders. Steps 6–7 turn the
// box into the neuron drawing; step 7 swaps the graph for stacked bars (two inputs).

import { useEffect, useState } from 'react';
import { bi, useT } from '../../i18n';
import { useLessonState } from '../../state/lessonState';
import { Plot2D, type PlotLayer } from '../../components/Plot2D';
import { Tex } from '../../components/Tex';
import { Slider } from '../../components/controls';
import { fmt, fmtSigned, tex } from '../../engine';
import type { StoryLesson } from '../types';
import { Appear, C, Knob, Machine, Note, Stage, ValueCard } from '../kit';
import { useTicker, useTween, useTweenArray } from '../useTween';
import { B0, B_SWEEP, guess, neuronOut, NOTEBOOKS, PENS, shares, stepTriangle, W0, W_PEN, W_SWEEP } from './math-neuron';

const ID = 'neuron';
const XD: [number, number] = [0, 4];
const YD: [number, number] = [-2.5, 14];
const W_MIN = 0;
const W_MAX = 5;
const B_MIN = -2;
const B_MAX = 3;
/** Where the step triangle sits: at the running example's input. */
const TRI_AT = NOTEBOOKS;

/** Larger plot text for video, scoped to this scene. */
const BIG_CSS = `.nn2 .plot2d .lbl{font-size:17px;font-weight:700}.nn2 .plot2d .ticks text{font-size:14px}.nn2 .axis-label{font-size:15px}.nn2-swap{animation:caption-in .5s ease}.nn2-sliders .slider label{font-size:1.25em;font-weight:700}.nn2-sliders .slider{gap:14px}`;

function Graph({ w, b, triangle }: { w: number; b: number; triangle: boolean }) {
  const { tr } = useT();
  const y = guess(w, b, NOTEBOOKS);
  const t = stepTriangle(w, b, TRI_AT);
  const layers: PlotLayer[] = [
    { kind: 'fn', f: (x) => guess(w, b, x), color: C.output, width: 4 },
    { kind: 'segment', from: [NOTEBOOKS, 0], to: [NOTEBOOKS, y], color: C.input, dash: '5 4', width: 2, opacity: 0.7 },
    { kind: 'segment', from: [0, y], to: [NOTEBOOKS, y], color: C.output, dash: '5 4', width: 2, opacity: 0.7 },
  ];
  if (triangle) {
    layers.push(
      { kind: 'segment', from: t.start, to: t.corner, color: C.input, width: 4, label: fmtSigned(t.run, 1) },
      { kind: 'segment', from: t.corner, to: t.end, color: C.slope, width: 4, label: fmtSigned(t.rise, 1) },
    );
  }
  layers.push({ kind: 'points', color: C.output, pts: [{ x: NOTEBOOKS, y, r: 8 }] });
  // Up and to the left of the dot, where the rising line is not.
  if (!triangle) layers.push({ kind: 'text', x: NOTEBOOKS - 0.12, y, dy: -12, anchor: 'end', color: C.output, text: `(${fmt(NOTEBOOKS, 1)}, ${fmt(y, 1)})` });
  return (
    <Plot2D
      xDomain={XD}
      yDomain={YD}
      layers={layers}
      height={360}
      xLabel={tr(bi('notebooks (x) →', 'تعداد دفترچه (x)'))}
      yLabel={tr(bi('price ↑', 'قیمت ↑'))}
      testId="neuron-graph"
    />
  );
}

/** The neuron drawing: input circle(s) → arrow with the weight → circle that adds b → ŷ. */
function NeuronDiagram({ two, out }: { two: boolean; out: number }) {
  const k = useTween(two ? 1 : 0, 700);
  const o = useTween(out, 700);
  const N = { x: 300, y: 150, r: 62 };
  const ins = [
    { y: 150 - 62 * k, x: NOTEBOOKS, w: W0, name: two ? 'x₁' : 'x', wn: two ? 'w₁' : 'w', op: 1 },
    { y: 150 + 70 * k, x: PENS, w: W_PEN, name: 'x₂', wn: 'w₂', op: k },
  ];
  const arrow = (x1: number, y1: number, x2: number, y2: number, color: string, key: string, op = 1) => {
    const d = Math.hypot(x2 - x1, y2 - y1);
    const ux = (x2 - x1) / d;
    const uy = (y2 - y1) / d;
    const hx = x2 - ux * 16;
    const hy = y2 - uy * 16;
    return (
      <g key={key} opacity={op}>
        <line x1={x1} y1={y1} x2={hx} y2={hy} stroke={color} strokeWidth={5} strokeLinecap="round" />
        <polygon points={`${x2},${y2} ${hx - uy * 9},${hy + ux * 9} ${hx + uy * 9},${hy - ux * 9}`} fill={color} />
      </g>
    );
  };
  return (
    <svg direction="ltr" viewBox="0 0 560 300" width="100%" style={{ maxWidth: 560 }} role="img" aria-label="neuron">
      {ins.map((p, i) => {
        const dx = N.x - 70;
        const dy = N.y - p.y;
        const d = Math.hypot(dx, dy);
        const ex = N.x - (dx / d) * (N.r + 4);
        const ey = N.y - (dy / d) * (N.r + 4);
        // The weight's label sits beside its arrow, above it for the top input and below for the second.
        const mx = 70 + 38 + (ex - 108) * 0.42;
        const my = p.y + (ey - p.y) * 0.42 + (i === 0 ? -18 : 34);
        return (
          <g key={i} opacity={p.op}>
            {arrow(70 + 38, p.y + (dy / d) * 10, ex, ey, C.weight, 'a' + i)}
            <text x={mx} y={my} textAnchor="middle" fontSize={22} fontWeight={700} fill={C.weight}>
              {p.wn} = {fmt(p.w, 2)}
            </text>
            <circle cx={70} cy={p.y} r={36} fill="var(--surface)" stroke={C.input} strokeWidth={4} />
            <text x={70} y={p.y + 11} textAnchor="middle" fontSize={32} fontWeight={700} fill={C.input}>
              {fmt(p.x, 2)}
            </text>
            <text x={70} y={p.y - 44} textAnchor="middle" fontSize={20} fontWeight={700} fill={C.input}>
              {p.name}
            </text>
          </g>
        );
      })}
      <circle cx={N.x} cy={N.y} r={N.r} fill="var(--surface-2)" stroke="var(--ink-2)" strokeWidth={4} />
      <text x={N.x} y={N.y + 2} textAnchor="middle" fontSize={34} fontWeight={700} fill={C.bias}>
        + b
      </text>
      <text x={N.x} y={N.y + 34} textAnchor="middle" fontSize={20} fontWeight={600} fill={C.bias}>
        b = {fmt(B0, 2)}
      </text>
      {arrow(N.x + N.r + 4, N.y, 450, N.y, C.output, 'out')}
      <circle cx={490} cy={N.y} r={38} fill="var(--surface)" stroke={C.output} strokeWidth={4} />
      <text x={490} y={N.y + 11} textAnchor="middle" fontSize={32} fontWeight={700} fill={C.output}>
        {fmt(o, 1)}
      </text>
      <text x={490} y={N.y - 48} textAnchor="middle" fontSize={22} fontWeight={700} fill={C.output}>
        ŷ
      </text>
    </svg>
  );
}

/** Two inputs: each input times its weight is a block; the blocks stack up to ŷ. */
function Bars() {
  const { tr } = useT();
  const [s1, s2] = shares([W0, W_PEN], [NOTEBOOKS, PENS]);
  const total = neuronOut([W0, W_PEN], B0, [NOTEBOOKS, PENS]);
  const [on, setOn] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setOn(true));
    return () => cancelAnimationFrame(id);
  }, []);
  const [h1, h2, hb] = useTweenArray([s1, on ? s2 : 0, on ? B0 : 0], 900);
  const H = 300;
  const k = (H - 40) / Math.max(10, total);
  const block = (bottom: number, h: number, color: string) => ({
    position: 'absolute' as const,
    left: 0,
    width: 120,
    bottom: bottom * k,
    height: Math.max(0, h * k),
    background: `color-mix(in srgb, ${color} 22%, transparent)`,
    border: `3px solid ${color}`,
    borderRadius: 6,
    boxSizing: 'border-box' as const,
  });
  const lbl = (bottom: number, color: string) => ({ position: 'absolute' as const, left: 140, bottom: bottom * k - 16, color, fontWeight: 700, fontSize: '1.35em', whiteSpace: 'nowrap' as const });
  const top = h1 + h2 + hb;
  return (
    <div style={{ position: 'relative', height: H, width: 420, margin: '0 auto' }} dir="ltr" data-testid="neuron-bars">
      <div style={{ position: 'absolute', left: -10, right: 0, bottom: 0, borderBottom: '2px solid var(--line-strong)' }} />
      <div style={block(0, h1, C.input)} />
      <div style={block(h1, h2, PEN_COLOR)} />
      <div style={lbl(h1 / 2, C.input)}>
        <Tex>{`\\cw{${tex(W0)}}\\cdot\\cx{${tex(NOTEBOOKS)}} = ${tex(s1)}`}</Tex>{' '}
        <span style={{ fontSize: '0.7em', color: 'var(--muted)' }}>{tr(bi('notebooks', 'دفترچه'))}</span>
      </div>
      <div style={{ ...lbl(h1 + h2 / 2, PEN_COLOR), opacity: on ? 1 : 0, transition: 'opacity .5s' }}>
        <Tex>{`\\cw{${tex(W_PEN)}}\\cdot\\cx{${tex(PENS)}} = ${tex(s2)}`}</Tex>{' '}
        <span style={{ fontSize: '0.7em', color: 'var(--muted)' }}>{tr(bi('pens', 'خودکار'))}</span>
      </div>
      <div style={{ position: 'absolute', left: -10, width: 140, bottom: top * k, borderTop: `3px dashed ${C.output}` }} />
      <div style={{ position: 'absolute', left: 0, width: 120, bottom: top * k + 6, textAlign: 'center', color: C.output, fontWeight: 700, fontSize: '1.6em' }}>
        ŷ = {fmt(top, 1)}
      </div>
    </div>
  );
}

/** The pens' block: a different colour from the notebooks' block (not a semantic colour). */
const PEN_COLOR = 'var(--c-wpos)';

const PIECES = ['\\cyhat{\\hat y}', '=', '\\cw{w}', '\\cdot', '\\cx{x}', '+', '\\cb{b}'];

/** ŷ = w·x + b, built piece by piece, then filled in with the running numbers. */
function FormulaBuild({ animate }: { animate: boolean }) {
  const tick = useTicker(animate, PIECES.length + 1, 420);
  const n = animate ? tick : PIECES.length + 1;
  const fill = `= \\cw{${tex(W0)}}\\cdot\\cx{${tex(NOTEBOOKS)}} + \\cb{${tex(B0)}} = \\cyhat{${tex(guess(W0, B0, NOTEBOOKS))}}`;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5em', fontSize: '2.1em' }} dir="ltr" data-testid="neuron-formula">
      <div style={{ display: 'flex', gap: '0.25em', alignItems: 'center' }}>
        {PIECES.map((p, i) => (
          <Appear key={i} show={n > i} keep>
            <Tex>{p}</Tex>
          </Appear>
        ))}
      </div>
      <Appear show={n > PIECES.length} keep>
        <Tex>{fill}</Tex>
      </Appear>
    </div>
  );
}

function Scene({ step }: { step: number }) {
  const { tr } = useT();
  const [s, set] = useLessonState(ID, { w: W0, b: B0 });
  const wTick = useTicker(step === 1, W_SWEEP.length - 1, 1300);
  const bTick = useTicker(step === 3, B_SWEEP.length - 1, 1300);
  const play = step === 7;
  const w = play ? s.w : step === 1 ? W_SWEEP[wTick] : W0;
  const b = play ? s.b : step === 3 ? B_SWEEP[bTick] : B0;
  const [wT, bT] = useTweenArray([w, b], 600);
  const yT = guess(wT, bT, NOTEBOOKS);
  // Step 1: the machine alone, centred; from step 2 it slides left to make room for the graph.
  const shift = useTween(step === 0 ? 1 : 0, 700);
  const diagram = step === 5 || step === 6;
  const showB = step >= 3;
  const rule = showB ? '\\cw{w}\\cdot\\cx{x} + \\cb{b}' : '\\cw{w}\\cdot\\cx{x}';

  return (
    <Stage>
      <style>{BIG_CSS}</style>
      <div className="nn2" dir="ltr" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 24, width: '100%' }}>
        <div style={{ flex: '0 0 auto', width: 560, display: 'flex', justifyContent: 'center', transform: `translateX(${shift * 272}px)` }}>
          {diagram ? (
            <div className="nn2-swap" style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <NeuronDiagram two={step === 6} out={step === 6 ? neuronOut([W0, W_PEN], B0, [NOTEBOOKS, PENS]) : guess(W0, B0, NOTEBOOKS)} />
              <Note>
                <span style={{ fontSize: '1.5em', fontWeight: 700, color: 'var(--ink)' }}>{tr(bi('a neuron', 'یک نورون'))}</span>
              </Note>
            </div>
          ) : (
            <div style={{ zoom: 0.92 }}>
              <Machine
                input={NOTEBOOKS}
                output={fmt(yT, 1)}
                rule={rule}
                pulse={step === 0 ? 0 : undefined}
                inputLabel={tr(bi('notebooks', 'دفترچه'))}
                outputLabel={tr(bi('price', 'قیمت'))}
                knobs={
                  <>
                    <Knob label="w" value={w} min={W_MIN} max={6} color={C.weight} size={84} digits={1} />
                    {showB && <Knob label="b" value={b} min={B_MIN} max={B_MAX} color={C.bias} size={84} digits={1} />}
                  </>
                }
              />
            </div>
          )}
        </div>
        <Appear show={step >= 1} keep style={{ flex: '0 0 auto', width: 500 }}>
          {step === 6 ? (
            <div className="nn2-swap">
              <Bars />
            </div>
          ) : (
            <Graph w={wT} b={bT} triangle={step === 2 || step === 3} />
          )}
        </Appear>
      </div>
      <div style={{ minHeight: 120, display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%' }}>
        {(step === 2 || step === 3) && (
          <div className="nn2-swap" style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
            <ValueCard label={tr(bi('slope', 'شیب'))} value={fmt(stepTriangle(wT, bT, TRI_AT).rise, 1)} color={C.slope} note={tr(bi('one more notebook → this much more', 'یک دفترچه بیشتر ← این‌قدر گران‌تر'))} />
            <span style={{ fontSize: '2em', color: 'var(--muted)' }}>=</span>
            <ValueCard label="w" value={fmt(wT, 1)} color={C.weight} />
          </div>
        )}
        {(step === 4 || step === 5) && <FormulaBuild animate={step === 4} />}
        {step === 6 && (
          <div className="nn2-swap" style={{ fontSize: '1.9em' }} dir="ltr">
            <Tex>{`\\cyhat{\\hat y} = \\cw{${tex(W0)}}\\cdot\\cx{${tex(NOTEBOOKS)}} + \\cw{${tex(W_PEN)}}\\cdot\\cx{${tex(PENS)}} + \\cb{${tex(B0)}} = \\cyhat{${tex(neuronOut([W0, W_PEN], B0, [NOTEBOOKS, PENS]))}}`}</Tex>
          </div>
        )}
        {play && (
          <div className="nn2-swap nn2-sliders" style={{ display: 'grid', gap: 10, width: 'min(560px, 100%)' }} data-testid="neuron-sliders">
            <Slider label="w" value={s.w} min={W_MIN} max={W_MAX} step={0.1} color="w" digits={2} onChange={(v) => set({ w: v })} testId="slider-w" />
            <Slider label="b" value={s.b} min={B_MIN} max={B_MAX} step={0.1} color="b" digits={2} onChange={(v) => set({ b: v })} testId="slider-b" />
          </div>
        )}
      </div>
    </Stage>
  );
}

export const neuronLesson: StoryLesson = {
  id: ID,
  Scene,
  steps: [
    {
      say: bi(
        'Our machine guesses prices: it takes a number of notebooks and guesses the total price. Inside it has a knob, w: the price of one notebook.',
        'ماشین ما قیمت حدس می‌زند: تعداد دفترچه را می‌گیرد و قیمت کل را حدس می‌زند. داخلش یک پیچ هست: w، قیمت هر دفترچه.',
      ),
      more: bi(
        'This is the machine from lesson 1, now with a real job: the number of notebooks goes in, the total price comes out. The 3 is no longer stuck inside the box; it sits on a purple knob called w. With w = 3, two notebooks cost 3 × 2 = 6 coins.',
        'این همان ماشین درس ۱ است، فقط حالا یک کار واقعی دارد: تعداد دفترچه‌ها وارد می‌شود و قیمت کل بیرون می‌آید. عدد ۳ دیگر داخل جعبه ثابت نیست؛ روی یک پیچ بنفش نشسته که اسمش w است. با w = ۳، دو دفترچه می‌شود ۳ × ۲ = ۶ سکه.',
      ),
    },
    {
      say: bi(
        'Turn the knob: a bigger w means each notebook costs more, and the line gets steeper.',
        'پیچ را بچرخان: w بزرگ‌تر یعنی هر دفترچه گران‌تر، و خط تندتر.',
      ),
      more: bi(
        'Watch the knob go from 3 to 5, then to 1, then back to 3. When w is 5, the same 2 notebooks cost 10, and the line on the right climbs faster. When w is 1, they cost only 2, and the line lies flatter.',
        'ببین پیچ از ۳ به ۵ می‌رود، بعد به ۱ و دوباره به ۳. وقتی w پنج است، همان ۲ دفترچه می‌شود ۱۰ و خط سمت راست تندتر بالا می‌رود. وقتی w یک است، فقط ۲ سکه می‌شود و خط خوابیده‌تر است.',
      ),
    },
    {
      say: bi(
        'The steepness of the line answers: one more notebook, how much more money? Exactly w. We call this the slope. Remember it.',
        'تندی خط یعنی: یک دفترچه بیشتر، قیمت چقدر بالا می‌رود؟ دقیقاً w. به این «شیب» می‌گوییم — یادت بماند.',
      ),
      more: bi(
        'Start at the dot on the line, take one step to the right (one more notebook), then climb back up to the line. The climb is exactly w: here 3 coins. It is the same climb anywhere on this line. This "how much up for one step right" is the slope, and it will come back many times in this course.',
        'از نقطه روی خط شروع کن، یک قدم به راست برو (یک دفترچه بیشتر) و بعد بالا برو تا دوباره به خط برسی. این بالا رفتن دقیقاً w است: این‌جا ۳ سکه. هر جای این خط هم همین‌قدر است. همین «یک قدم به راست، چقدر بالا؟» اسمش شیب است و در این دوره بارها برمی‌گردد.',
      ),
    },
    {
      say: bi(
        'A second knob: b, a fixed fee like delivery. It moves the line up and down without changing its steepness.',
        'پیچ دوم: b، یک هزینه ثابت مثل پیک. خط را بالا و پایین می‌برد ولی تندی‌اش را عوض نمی‌کند.',
      ),
      more: bi(
        'b is added to every price, whatever the number of notebooks, like a delivery fee of 2 coins. Adding the same amount everywhere lifts the whole line without tilting it. That is why the step triangle stays the same size: one more notebook still costs exactly w more.',
        'b به هر قیمتی اضافه می‌شود، هر چند تا دفترچه که باشد؛ مثل ۲ سکه هزینه پیک. وقتی به همه جا یک مقدار ثابت اضافه کنی، کل خط بالا می‌رود ولی کج نمی‌شود. برای همین مثلث پله همان اندازه می‌ماند: یک دفترچه بیشتر هنوز دقیقاً w گران‌تر است.',
      ),
    },
    {
      say: bi('So the machine computes $\\hat y = w \\cdot x + b$. The hat on y means "a guess".', 'پس کار ماشین این است: $\\hat y = w \\cdot x + b$. کلاه روی y یعنی «حدس».'),
      more: bi(
        'Read it as: "the guess is w times x, plus b". x is the input (the notebooks), w and b are the two knobs, and $\\hat y$ ("y hat") is what the machine says. With our numbers: $3 \\cdot 2 + 0 = 6$.',
        'این‌طور بخوانش: «حدس برابر است با w ضرب در x، به‌علاوه b». x ورودی است (تعداد دفترچه)، w و b دو پیچ ماشین‌اند و $\\hat y$ («وای کلاه») همان چیزی است که ماشین می‌گوید. با عددهای ما: $3 \\cdot 2 + 0 = 6$.',
      ),
    },
    {
      say: bi('We call this little machine a neuron: one input, one weight, one bias, one output.', 'این ماشین کوچک را «نورون» می‌نامیم: یک ورودی، یک وزن، یک بایاس، یک خروجی.'),
      more: bi(
        'This is how a neuron is usually drawn: the input on the left, an arrow carrying w, a circle that adds b, and the guess $\\hat y$ on the right. "Weight" is just the name of the w knob and "bias" the name of the b knob. It is the same machine as before, drawn differently.',
        'نورون را معمولاً این‌طور می‌کشند: ورودی در چپ، یک فلش که w را با خودش دارد، یک دایره که b را اضافه می‌کند، و حدس $\\hat y$ در راست. «وزن» فقط اسم پیچ w است و «بایاس» اسم پیچ b. همان ماشین قبلی است، فقط جور دیگری کشیده شده.',
      ),
    },
    {
      say: bi('With several inputs, each input has its own weight, and everything is added up.', 'اگر چند ورودی داشته باشد، هر ورودی وزن خودش را دارد و همه با هم جمع می‌شوند.'),
      more: bi(
        'Now the shop also sells pens. $x_1$ is the number of notebooks with its weight $w_1 = 3$, and $x_2$ is the number of pens with its weight $w_2 = 2$; the small 1 and 2 only say which input. Each input times its own weight is one block of the price. The blocks stack up and b is added: $3 \\cdot 2 + 2 \\cdot 1 + 0 = 8$.',
        'حالا مغازه خودکار هم می‌فروشد. $x_1$ تعداد دفترچه است با وزن خودش $w_1 = 3$، و $x_2$ تعداد خودکار با وزن $w_2 = 2$؛ عددهای کوچک ۱ و ۲ فقط می‌گویند کدام ورودی. هر ورودی ضرب در وزن خودش یک تکه از قیمت است. تکه‌ها روی هم جمع می‌شوند و b هم اضافه می‌شود: $3 \\cdot 2 + 2 \\cdot 1 + 0 = 8$.',
      ),
    },
    {
      say: bi(
        "Learning means finding the right numbers for these knobs. First, let's see what happens when we connect neurons.",
        'یاد گرفتن یعنی پیدا کردن عددهای درست برای این پیچ‌ها. ولی قبلش ببینیم با وصل کردن نورون‌ها چه می‌شود.',
      ),
      more: bi(
        'Move the two sliders yourself. w tilts the line, b lifts or lowers it, and the price for 2 notebooks follows along. Later, the computer will turn these knobs by itself until its guesses are right. That is what "learning" will mean.',
        'خودت دو اسلایدر را جابه‌جا کن. w خط را کج می‌کند، b آن را بالا و پایین می‌برد و قیمت ۲ دفترچه هم همراهش عوض می‌شود. بعداً خود کامپیوتر این پیچ‌ها را می‌چرخاند تا حدس‌هایش درست شود. «یاد گرفتن» یعنی همین.',
      ),
    },
  ],
};
