// Lesson 3 · Bending and building a network. ReLU bends the line; neurons in layers; the forward pass.
// Script: docs/storyboard.md, lesson 3.
//
// One scene that grows: the two-neuron chain (steps 1–3, with its graph underneath), one bent
// neuron (4), four bent lines adding up to a U (5), the network that drew it, in layers (6), and
// the real chain network of preset chain-two in the lab canvas (7–8). Loss and gradients stay
// hidden: they belong to lesson 4.

import { useEffect, useLayoutEffect, useState } from 'react';
import { bi, useT } from '../../i18n';
import { useLab, useLabDerived } from '../../state/lab';
import { NetworkCanvas } from '../../components/NetworkCanvas';
import { Plot2D, type PlotLayer } from '../../components/Plot2D';
import { Tex } from '../../components/Tex';
import { fmt, tex, texP } from '../../engine';
import type { StoryLesson } from '../types';
import { Appear, C, Note, Stage, ValueCard } from '../kit';
import { useTicker, useTween } from '../useTween';
import { bendAt, bentNeuron, chainOut, chainTwo, chainValues, collapsed, HINGES, piece, relu } from './math-network';

const ID = 'network';
/** The running numbers, from preset chain-two itself (x = 2, w₁ = 3, w₂ = 2, biases 0). */
const P = chainTwo();
/** Activation colour (what comes out of a bend), as on the network canvas. */
const ACT = 'var(--c-a)';
/** Step 4's single bent neuron. */
const BENT = { w: P.w1, b: -2 };
/** The four bent lines of step 5 (not semantic colours, just four different ones). */
const HC = ['var(--c-a)', 'var(--c-wneg)', 'var(--c-wpos)', '#8a63d2'];

const BIG_CSS = `.nn3 .plot2d .lbl{font-size:17px;font-weight:700}.nn3 .plot2d .ticks text{font-size:14px}.nn3 .axis-label{font-size:15px}.nn3-in{animation:caption-in .5s ease}.nn3 .nn-canvas{border:1px solid var(--line);border-radius:12px}.nn3 .nn-unit{font-size:21px;width:150px;padding-top:14px}.nn3 .nn-unit .nn-act{font-size:15px}.nn3 .nn-wlabel{font-size:20px;font-weight:700}.nn3 .react-flow__controls{display:none}`;

/** False on the first frame, true after: lets a freshly shown element animate from its start. */
function useEntered() {
  const [on, setOn] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setOn(true));
    return () => cancelAnimationFrame(id);
  }, []);
  return on;
}

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

// ---------------------------------------------------------------- the chain of two neurons

const CX = { in: 90, n1: 400, n2: 710 };
const CY = 105;

function Arrow({ x1, x2, y, color, label }: { x1: number; x2: number; y: number; color: string; label?: string }) {
  return (
    <g>
      <line x1={x1} y1={y} x2={x2 - 16} y2={y} stroke={color} strokeWidth={5} strokeLinecap="round" />
      <polygon points={`${x2},${y} ${x2 - 18},${y - 10} ${x2 - 18},${y + 10}`} fill={color} />
      {label && (
        <text x={(x1 + x2) / 2} y={y - 18} textAnchor="middle" fontSize={30} fontWeight={700} fill={color}>
          {label}
        </text>
      )}
    </g>
  );
}

/** A small hinge icon: flat, then rising. */
const hingePath = (x: number, y: number, s: number) => `M${x - s},${y + s * 0.5} L${x},${y + s * 0.5} L${x + s},${y - s * 0.6}`;

/**
 * Input → neuron ×w₁ → neuron ×w₂. With `travel`, a dot carries the number from left to right and
 * each neuron's result appears as the dot arrives. With `bend`, a ReLU sits between the neurons.
 */
function ChainDiagram({ travel, bend, maxWidth }: { travel: boolean; bend: boolean; maxWidth: number }) {
  const tick = useTicker(travel, 2, 1300);
  const stops = [CX.in, CX.n1, CX.n2];
  const tx = useTween(travel ? stops[tick] : CX.n2, 1000);
  const [x, v1, v2] = chainValues(P.w1, P.b1, P.w2, P.b2, bend ? 'relu' : 'linear', P.x);
  const seen1 = travel ? clamp01((tx - (CX.n1 - 60)) / 60) : 1;
  const seen2 = travel ? clamp01((tx - (CX.n2 - 60)) / 60) : 1;
  const moving = travel && Math.abs(tx - stops[tick]) > 2;
  const k = useTween(bend ? 1 : 0, 600);
  const mid = (CX.n1 + CX.n2) / 2;
  return (
    <svg direction="ltr" viewBox="0 0 800 190" width="100%" style={{ maxWidth }} role="img" aria-label="two neurons in a row" data-testid="chain-diagram">
      <Arrow x1={CX.in + 48} x2={CX.n1 - 58} y={CY} color={C.weight} label={`×${fmt(P.w1, 2)}`} />
      <Arrow x1={CX.n1 + 58} x2={CX.n2 - 58} y={CY} color={C.weight} label={`×${fmt(P.w2, 2)}`} />
      <g opacity={k}>
        <rect x={mid - 52} y={CY + 16} width={104} height={44} rx={10} fill="var(--surface)" stroke={ACT} strokeWidth={3} />
        <path d={hingePath(mid - 22, CY + 38, 13)} stroke={ACT} strokeWidth={4} fill="none" strokeLinejoin="round" />
        <text x={mid + 18} y={CY + 45} textAnchor="middle" fontSize={18} fontWeight={700} fill={ACT}>
          ReLU
        </text>
      </g>
      <circle cx={CX.in} cy={CY} r={46} fill="var(--surface)" stroke={C.input} strokeWidth={4} />
      <text x={CX.in} y={CY + 13} textAnchor="middle" fontSize={38} fontWeight={700} fill={C.input}>
        {fmt(x, 2)}
      </text>
      <text x={CX.in} y={CY - 58} textAnchor="middle" fontSize={22} fontWeight={700} fill={C.input}>
        x
      </text>
      <circle cx={CX.n1} cy={CY} r={56} fill="var(--surface-2)" stroke="var(--ink-2)" strokeWidth={4} />
      <text x={CX.n1} y={CY + 13} textAnchor="middle" fontSize={38} fontWeight={700} fill="var(--ink)" opacity={seen1}>
        {fmt(v1, 2)}
      </text>
      <circle cx={CX.n2} cy={CY} r={56} fill="var(--surface-2)" stroke={C.output} strokeWidth={4} />
      <text x={CX.n2} y={CY + 13} textAnchor="middle" fontSize={38} fontWeight={700} fill={C.output} opacity={seen2}>
        {fmt(v2, 2)}
      </text>
      <text x={CX.n2} y={CY - 68} textAnchor="middle" fontSize={22} fontWeight={700} fill={C.output}>
        ŷ
      </text>
      {moving && <circle cx={tx} cy={CY} r={13} fill={C.input} stroke="var(--surface)" strokeWidth={3} />}
    </svg>
  );
}

// ---------------------------------------------------------------- graphs

/** Step 2: the chain's output for every x is one straight line, the same as one neuron ×6. */
function ChainLineGraph() {
  const { tr } = useT();
  const one = collapsed(P.w1, P.b1, P.w2, P.b2);
  const y = chainOut(P.w1, P.b1, P.w2, P.b2, 'linear', P.x);
  const layers: PlotLayer[] = [
    { kind: 'fn', f: (x) => chainOut(P.w1, P.b1, P.w2, P.b2, 'linear', x), color: C.output, width: 7, opacity: 0.35 },
    { kind: 'fn', f: (x) => one.w * x + one.b, color: C.output, width: 3, dash: '10 7' },
    { kind: 'points', color: C.output, pts: [{ x: P.x, y, r: 8 }] },
    { kind: 'text', x: P.x - 0.12, y, dy: -10, anchor: 'end', color: C.output, text: `(${fmt(P.x, 2)}, ${fmt(y, 2)})` },
  ];
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 28, width: '100%', justifyContent: 'center' }}>
      <div style={{ width: 560 }}>
        <Plot2D xDomain={[-2, 3]} yDomain={[-13, 19]} layers={layers} height={300} xLabel={tr(bi('x →', 'x →'))} yLabel={tr(bi('chain output ↑', 'خروجی زنجیره ↑'))} testId="chain-line" />
      </div>
      <div style={{ display: 'grid', gap: 10, justifyItems: 'center' }}>
        <div style={{ fontSize: '2em' }} dir="ltr">
          <Tex>{`\\cw{\\times ${tex(P.w1)}}\\;\\to\\;\\cw{\\times ${tex(P.w2)}} \\;=\\; \\cw{\\times ${tex(one.w)}}`}</Tex>
        </div>
        <div style={{ fontSize: '1.3em', fontWeight: 700, color: C.output, display: 'flex', gap: 8, alignItems: 'baseline' }}>
          <span>{tr(bi('same as one neuron', 'همان یک نورون'))}</span>
          <Tex>{`\\times ${tex(one.w)}`}</Tex>
        </div>
        <Note>{tr(bi('still a straight line', 'باز هم یک خط راست'))}</Note>
      </div>
    </div>
  );
}

const RELU_XS = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5];

/** Step 3: the ReLU graph with a dot sliding back and forth along it. */
function ReluGraph() {
  const { tr } = useT();
  const tick = useTicker(true, 10000, 1000);
  const n = RELU_XS.length - 1;
  const i = tick % (2 * n);
  const x = useTween(RELU_XS[i <= n ? i : 2 * n - i], 900);
  const y = relu(x);
  const layers: PlotLayer[] = [
    { kind: 'fn', f: relu, color: ACT, width: 5 },
    { kind: 'segment', from: [x, 0], to: [x, y], color: C.input, dash: '5 4', width: 2.5 },
    { kind: 'points', color: ACT, pts: [{ x, y, r: 9 }] },
    { kind: 'points', color: C.input, pts: [{ x, y: 0, r: 6, hollow: true }] },
  ];
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 28, width: '100%', justifyContent: 'center' }}>
      <div style={{ width: 520 }}>
        <Plot2D xDomain={[-3, 3]} yDomain={[-1, 3.2]} layers={layers} height={290} xLabel={tr(bi('input →', 'ورودی'))} yLabel={tr(bi('after ReLU ↑', 'بعد از ReLU ↑'))} testId="relu-graph" />
      </div>
      <div style={{ display: 'grid', gap: 10, justifyItems: 'center' }}>
        <ValueCard label={tr(bi('in', 'ورودی'))} value={fmt(x, 1)} color={C.input} />
        <div style={{ fontSize: '1.8em', color: ACT }}>↓ ReLU</div>
        <ValueCard label={tr(bi('out', 'خروجی'))} value={fmt(y, 1)} color={ACT} />
      </div>
    </div>
  );
}

/** Step 4: one neuron with w = 3, b = −2. The straight line folds up where it would go below 0. */
function BentGraph() {
  const { tr } = useT();
  const on = useEntered();
  const k = useTween(on ? 1 : 0, 1200);
  const at = bendAt(BENT.w, BENT.b) ?? 0;
  const line = (x: number) => BENT.w * x + BENT.b;
  // Halfway through the fold the negative part is lifted halfway; at the end it is ReLU exactly.
  const folding = (x: number) => (k >= 1 ? bentNeuron(BENT.w, BENT.b, x) : line(x) >= 0 ? line(x) : (1 - k) * line(x));
  const layers: PlotLayer[] = [
    { kind: 'fn', f: line, color: 'var(--ink-2)', width: 2.5, dash: '7 6', opacity: 0.45 },
    { kind: 'fn', f: folding, color: ACT, width: 5.5 },
    { kind: 'points', color: ACT, pts: [{ x: at, y: 0, r: 8 }] },
    { kind: 'text', x: at + 0.08, y: 0, dy: 30, anchor: 'start', color: ACT, text: `x = ${fmt(at, 2)}` },
  ];
  return (
    <div style={{ width: 'min(640px, 100%)' }}>
      <Plot2D xDomain={[-1, 2.5]} yDomain={[-5.5, 6]} layers={layers} height={330} xLabel={tr(bi('x →', 'x →'))} yLabel={tr(bi('output ↑', 'خروجی ↑'))} testId="bent-graph" />
    </div>
  );
}

/** Step 5: four bent lines appear one at a time; their sum (thick) traces a U. */
function HingeGraph() {
  const { tr } = useT();
  const tick = useTicker(true, HINGES.length, 1300);
  const t = useTween(tick, 900);
  const share = (i: number) => clamp01(t - i);
  const sum = (x: number) => HINGES.reduce((s, u, i) => s + share(i) * piece(u, x), 0);
  const layers: PlotLayer[] = [
    ...HINGES.map((u, i): PlotLayer => ({ kind: 'fn', f: (x) => piece(u, x), color: HC[i], width: 3, opacity: share(i) * 0.9 })),
    { kind: 'fn', f: sum, color: C.output, width: 6.5 },
  ];
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 28, width: '100%', justifyContent: 'center' }}>
      <div style={{ width: 600 }}>
        <Plot2D xDomain={[-2.5, 2.5]} yDomain={[-0.5, 6.5]} layers={layers} height={360} xLabel={tr(bi('x →', 'x →'))} yLabel={tr(bi('sum ↑', 'جمع ↑'))} testId="hinge-graph" />
      </div>
      <div style={{ display: 'grid', gap: 10, justifyItems: 'start' }}>
        {HINGES.map((_, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, opacity: 0.15 + 0.85 * share(i) }}>
            <svg direction="ltr" width={44} height={30} aria-hidden="true">
              <path d={hingePath(22, 16, 16)} stroke={HC[i]} strokeWidth={4} fill="none" transform={HINGES[i].w < 0 ? 'translate(44,0) scale(-1,1)' : undefined} />
            </svg>
            <span className="num" style={{ fontSize: '1.4em', fontWeight: 700, color: HC[i] }}>
              {i + 1}
            </span>
          </div>
        ))}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, borderTop: '2px solid var(--line-strong)', paddingTop: 8 }}>
          <span style={{ width: 44, height: 6, background: C.output, borderRadius: 3 }} />
          <span style={{ fontSize: '1.3em', fontWeight: 700, color: C.output }}>{tr(bi('sum', 'جمع'))}</span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- layers

/** Step 6: the network that drew the U, in three labelled columns. */
function LayersDiagram() {
  const { tr } = useT();
  const tick = useTicker(true, 3, 700);
  const cols = [100, 300, 500];
  const hy = [45, 125, 205, 285];
  const shown = (c: number) => tick > c;
  return (
    <div style={{ width: 'min(640px, 100%)' }} data-testid="layers-diagram">
      <svg direction="ltr" viewBox="0 0 600 330" width="100%" role="img" aria-label="layers">
        {hy.map((y, i) => (
          <g key={i} style={{ transition: 'opacity .5s' }} opacity={shown(1) ? 0.7 : 0}>
            <line x1={cols[0] + 40} y1={165} x2={cols[1] - 34} y2={y} stroke={C.weight} strokeWidth={3} />
          </g>
        ))}
        {hy.map((y, i) => (
          <g key={i} style={{ transition: 'opacity .5s' }} opacity={shown(2) ? 0.7 : 0}>
            <line x1={cols[1] + 34} y1={y} x2={cols[2] - 40} y2={165} stroke={C.weight} strokeWidth={3} />
          </g>
        ))}
        <g style={{ transition: 'opacity .5s' }} opacity={shown(0) ? 1 : 0}>
          <circle cx={cols[0]} cy={165} r={40} fill="var(--surface)" stroke={C.input} strokeWidth={4} />
          <text x={cols[0]} y={175} textAnchor="middle" fontSize={28} fontWeight={700} fill={C.input}>
            x
          </text>
        </g>
        <g style={{ transition: 'opacity .5s' }} opacity={shown(1) ? 1 : 0}>
          {hy.map((y, i) => (
            <g key={i}>
              <circle cx={cols[1]} cy={y} r={34} fill="var(--surface-2)" stroke={HC[i]} strokeWidth={4} />
              <path d={hingePath(cols[1], y + 2, 14)} stroke={HC[i]} strokeWidth={4} fill="none" transform={HINGES[i].w < 0 ? `translate(${2 * cols[1]},0) scale(-1,1)` : undefined} />
            </g>
          ))}
        </g>
        <g style={{ transition: 'opacity .5s' }} opacity={shown(2) ? 1 : 0}>
          <circle cx={cols[2]} cy={165} r={40} fill="var(--surface)" stroke={C.output} strokeWidth={4} />
          <text x={cols[2]} y={175} textAnchor="middle" fontSize={28} fontWeight={700} fill={C.output}>
            ŷ
          </text>
        </g>
      </svg>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', textAlign: 'center', fontWeight: 700, fontSize: '1.3em' }} dir="ltr">
        <Appear show={shown(0)} keep>
          <span style={{ color: C.input }}>{tr(bi('input layer', 'لایه ورودی'))}</span>
        </Appear>
        <Appear show={shown(1)} keep>
          <span>{tr(bi('hidden layer', 'لایه پنهان'))}</span>
        </Appear>
        <Appear show={shown(2)} keep>
          <span style={{ color: C.output }}>{tr(bi('output layer', 'لایه خروجی'))}</span>
        </Appear>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- the forward pass (lab canvas)

/**
 * Steps 7–8: preset chain-two in the lab canvas. The lab's forward stepper reveals one layer at a
 * time (k = 1 input … L + 1 output); it never reaches the loss, so no loss or gradient shows.
 */
function ForwardPass({ play }: { play: boolean }) {
  const { tr } = useT();
  const L = useLab((s) => s.net.layers.length);
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const der = useLabDerived();
  const tick = useTicker(play, L, 1300);
  // k = L + 2 is "every forward value shown, nothing active"; the loss stays hidden (showLoss off).
  const k = play ? tick + 1 : L + 2;
  // Before paint, so the canvas never shows the "everything" state (which includes gradients).
  useLayoutEffect(() => {
    useLab.setState({ stepper: { phase: 'forward', k } });
  }, [k]);
  useEffect(() => () => useLab.getState().showAll(), []);
  const tr0 = der.one.trace;
  const yhat = tr0.a[L][0];
  const target = data.Y[der.sample][0];
  const lines = [
    `\\cx{x} = \\cx{${tex(tr0.a[0][0])}}`,
    ...net.layers.map((layer, l) => {
      const input = l === 0 ? `\\cx{${texP(tr0.a[0][0])}}` : texP(tr0.a[l][0]);
      const out = tr0.a[l + 1][0];
      const last = l === L - 1;
      const sum = `\\cw{${texP(layer.W[0][0])}}\\cdot ${input} + \\cb{${texP(layer.b[0])}} = ${tex(tr0.z[l][0])}`;
      const after = last ? `\\cyhat{${tex(out)}}` : tex(out);
      return layer.act === 'relu' ? `${sum} \\;\\to\\; \\relu(${tex(tr0.z[l][0])}) = ${after}` : `${sum} = ${after}`;
    }),
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, width: '100%' }}>
      <div style={{ width: 'min(860px, 100%)' }}>
        <NetworkCanvas height={play ? 300 : 280} showLoss={false} interactive={false} />
      </div>
      {play ? (
        <div style={{ display: 'grid', gap: 6, fontSize: '1.55em', minHeight: 150 }} dir="ltr" data-testid="forward-lines">
          {lines.map((s, i) => (
            <Appear key={i} show={k > i} keep>
              <Tex>{s}</Tex>
            </Appear>
          ))}
        </div>
      ) : (
        <div className="nn3-in" style={{ display: 'flex', alignItems: 'center', gap: 22 }} dir="ltr" data-testid="guess-vs-real">
          <ValueCard big label={tr(bi('network says ŷ', 'شبکه می‌گوید ŷ'))} value={fmt(yhat, 2)} color={C.output} />
          <span style={{ fontSize: '4em', fontWeight: 800, color: C.loss }}>?</span>
          <ValueCard big label={tr(bi('real price', 'قیمت واقعی'))} value={fmt(target, 2)} color={C.target} />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- scene

function Scene({ step }: { step: number }) {
  const { tr } = useT();
  let body;
  if (step <= 2) {
    body = (
      <>
        <ChainDiagram key={step === 0 ? 'travel' : 'still'} travel={step === 0} bend={step === 2} maxWidth={step === 0 ? 900 : 620} />
        {step === 1 && (
          <div className="nn3-in" style={{ width: '100%' }}>
            <ChainLineGraph />
          </div>
        )}
        {step === 2 && (
          <div className="nn3-in" style={{ width: '100%' }}>
            <ReluGraph />
          </div>
        )}
      </>
    );
  } else if (step === 3) {
    body = (
      <div className="nn3-in" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, width: '100%' }}>
        <div style={{ fontSize: '1.9em' }} dir="ltr">
          <Tex>{`\\cyhat{\\hat y} = \\relu(\\cw{${tex(BENT.w)}}\\cdot\\cx{x} + \\cb{${texP(BENT.b)}})`}</Tex>
        </div>
        <BentGraph />
        <Note>{tr(bi('faint: before the bend · thick: after ReLU', 'کم‌رنگ: قبل از خم · پررنگ: بعد از ReLU'))}</Note>
      </div>
    );
  } else if (step === 4) {
    body = (
      <div className="nn3-in" style={{ width: '100%' }}>
        <HingeGraph />
      </div>
    );
  } else if (step === 5) {
    body = (
      <div className="nn3-in" style={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
        <LayersDiagram />
      </div>
    );
  } else {
    body = <ForwardPass play={step === 6} />;
  }
  return (
    <Stage>
      <style>{BIG_CSS}</style>
      <div className="nn3" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14, width: '100%' }}>
        {body}
      </div>
    </Stage>
  );
}

export const networkLesson: StoryLesson = {
  id: ID,
  Scene,
  presetId: 'chain-two',
  steps: [
    {
      say: bi(
        "Put two neurons in a row: the first one's output is the second one's input. 2 → ×3 → 6 → ×2 → 12.",
        'دو نورون را پشت سر هم بگذار: خروجی اولی، ورودی دومی است. ۲ ← ×۳ ← ۶ ← ×۲ ← ۱۲.',
      ),
      more: bi(
        'The 2 goes into the first neuron, which multiplies it by its weight 3 and adds its bias 0: out comes 6. That 6 is now the input of the second neuron, which multiplies it by 2: out comes 12. Each circle is the same little machine from lesson 2, with its own knobs.',
        'عدد ۲ وارد نورون اول می‌شود؛ نورون آن را در وزنش یعنی ۳ ضرب می‌کند و بایاسش را که صفر است اضافه می‌کند: ۶ بیرون می‌آید. حالا همین ۶ ورودیِ نورون دوم است که آن را در ۲ ضرب می‌کند: ۱۲ بیرون می‌آید. هر دایره همان ماشین کوچک درس ۲ است، با پیچ‌های خودش.',
      ),
    },
    {
      say: bi(
        'One problem: two straight lines in a row are still a straight line. ×3 then ×2 is just ×6.',
        'یک مشکل: دو خط راست پشت سر هم، باز هم یک خط راست است. ×۳ و بعد ×۲ همان ×۶ است.',
      ),
      more: bi(
        'The graph shows what the chain gives for every input x. It is a straight line: exactly the line that a single neuron with w = 6 draws (the dashed line on top of it). However many plain neurons you chain, you still get one straight line, and most things in the world are not straight lines.',
        'نمودار نشان می‌دهد این زنجیره برای هر ورودی x چه می‌دهد. یک خط راست است: دقیقاً همان خطی که یک نورون تنها با w = ۶ می‌کشد (خط‌چین روی آن). هر چند تا نورون ساده را پشت هم بگذاری، باز فقط یک خط راست داری؛ و بیشتر چیزهای دنیا خط راست نیستند.',
      ),
    },
    {
      say: bi(
        'The fix: put a bend between them. ReLU: turn negative numbers into 0, leave positive numbers alone.',
        'راه‌حل: بین‌شان یک «خم» بگذار. ReLU: عدد منفی را صفر کن، عدد مثبت را دست نزن.',
      ),
      more: bi(
        'ReLU is a tiny rule that sits right after a neuron. Watch the dot: while the input is negative, the output stays at 0 (the flat part). Once the input is positive, the output is the input itself (the rising part). The corner at 0 is the bend.',
        'ReLU یک قاعده خیلی ساده است که درست بعد از نورون می‌نشیند. نقطه را نگاه کن: تا وقتی ورودی منفی است، خروجی صفر می‌ماند (بخش صاف). وقتی ورودی مثبت شد، خروجی خودِ ورودی است (بخش بالارونده). آن گوشه روی صفر همان «خم» است.',
      ),
    },
    {
      say: bi('Now the neuron\'s output is a bent line: flat in one part, rising in the other.', 'حالا خروجی نورون یک خط شکسته است: یک‌جا صاف، یک‌جا بالارونده.'),
      more: bi(
        'Take one neuron with w = 3 and b = −2 and put ReLU after it. The faint line is $3x - 2$, before the bend. Wherever it would drop below 0, ReLU folds it up to 0. The corner sits where $3x - 2$ is exactly 0, at about $x = 0.67$; changing b slides that corner left or right.',
        'یک نورون با w = ۳ و b = −۲ بگیر و بعدش ReLU بگذار. خط کم‌رنگ $3x - 2$ است، قبل از خم. هر جا که این خط بخواهد زیر صفر برود، ReLU آن را تا صفر بالا می‌آورد. گوشه جایی است که $3x - 2$ دقیقاً صفر می‌شود، حدود $x = 0.67$؛ با عوض کردن b این گوشه به چپ و راست می‌رود.',
      ),
    },
    {
      say: bi(
        'Add a few of these bent lines and you can build almost any shape. That is where a network\'s power comes from.',
        'چند تا از این خط‌های شکسته را جمع کن، تقریباً هر شکلی می‌سازی. قدرت شبکه از همین‌جاست.',
      ),
      more: bi(
        'Each thin coloured line is one neuron with ReLU, each with its corner in a different place. Adding them up point by point gives the thick green curve. With only 4 bent lines we already get a U shape; with more neurons there are more corners, and the sum can follow almost any curve.',
        'هر خط نازک رنگی یک نورون با ReLU است و گوشه هر کدام جای دیگری است. اگر نقطه‌به‌نقطه جمع‌شان کنی، منحنی سبز پررنگ ساخته می‌شود. فقط با ۴ خط شکسته یک شکل U داریم؛ با نورون‌های بیشتر، گوشه‌های بیشتری داریم و جمع‌شان تقریباً هر منحنی‌ای را می‌تواند دنبال کند.',
      ),
    },
    {
      say: bi('We arrange neurons in columns; each column is a layer.', 'نورون‌ها را ستون‌ستون می‌چینیم؛ هر ستون یک «لایه» است.'),
      more: bi(
        'This is the very network that drew the U: one input, 4 bent neurons in the middle, and one output neuron that adds them up. The middle column is called a hidden layer, because from outside you only see the input and the output. Every line between two circles carries its own weight.',
        'این دقیقاً همان شبکه‌ای است که U را کشید: یک ورودی، ۴ نورون خم‌دار در وسط، و یک نورون خروجی که آن‌ها را جمع می‌کند. به ستون وسط «لایه پنهان» می‌گویند، چون از بیرون فقط ورودی و خروجی دیده می‌شود. هر خطی که دو دایره را به هم وصل می‌کند وزن خودش را دارد.',
      ),
    },
    {
      say: bi(
        'A number enters on the left and moves forward layer by layer to the output. This is the forward pass.',
        'عدد از چپ وارد می‌شود و لایه‌به‌لایه جلو می‌رود تا خروجی. به این «انتشار رو به جلو» می‌گوییم.',
      ),
      more: bi(
        'Back to our two-neuron chain, now with ReLU in each neuron. Each circle shows two numbers: z is $w \\cdot x + b$ before the bend, and a is what comes out after ReLU. Layer by layer: 2 goes in, the first neuron gives 6, the second gives 12. The last one is the guess $\\hat y$.',
        'برگردیم به زنجیره دونورونی خودمان، این بار با ReLU در هر نورون. هر دایره دو عدد نشان می‌دهد: z همان $w \\cdot x + b$ است قبل از خم، و a چیزی است که بعد از ReLU بیرون می‌آید. لایه‌به‌لایه: ۲ وارد می‌شود، نورون اول ۶ می‌دهد و دومی ۱۲. آخری همان حدس $\\hat y$ است.',
      ),
    },
    {
      say: bi('For 2 notebooks our network says 12. But the real price was 10. How wrong is it?', 'شبکه ما برای ۲ دفترچه می‌گوید ۱۲. ولی قیمت واقعی ۱۰ بود. چقدر اشتباه کرده؟'),
      more: bi(
        'The network guesses 12 coins, but 2 notebooks really cost 10. So the guess is too high. To make it better we first need one number that says how bad a guess is. That number is the topic of the next lesson.',
        'شبکه ۱۲ سکه حدس می‌زند، ولی ۲ دفترچه در واقع ۱۰ سکه است. پس حدسش زیادی بالاست. برای بهتر کردنش اول یک عدد لازم داریم که بگوید حدس چقدر بد است. موضوع درس بعد همین عدد است.',
      ),
    },
  ],
};
