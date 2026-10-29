// Lesson 10: Chain rule on the core teaching example (x=2, w1=3, w2=2, b=0, ReLU, y=10).
// Everything is read from the shared lab store (preset "chain-two"), so the network, the
// factor cards, the equations and the plots always show the same numbers.

import { useMemo } from 'react';
import type { LessonDef } from './types';
import { bi, useT } from '../i18n';
import { useLab, useLabDerived } from '../state/lab';
import { useLessonState } from '../state/lessonState';
import { NetworkCanvas } from '../components/NetworkCanvas';
import { Plot2D } from '../components/Plot2D';
import { Tex } from '../components/Tex';
import { Segmented, Slider, Stat } from '../components/controls';
import { Advanced, Callout, Cycle, P, Predict, Reveal } from '../components/teaching';
import { actDeriv, fmt, fmtSigned, lossSample, predict, setParam, tex, texP, type Network } from '../engine';
import { useUi } from '../state/ui';

const ID = 'l10';

type Factor = 'dL' | 'dout' | 'da1';
interface S {
  focus: Factor;
  target: 'w1' | 'w2';
  delta: number;
  /** How many multiplications of the composition animation are shown (0-3). */
  composed: number;
}
const INIT: S = { focus: 'dL', target: 'w1', delta: 0.01, composed: 3 };
const useS = () => useLessonState<S>(ID, INIT);

/** All forward and backward quantities of the two-neuron chain, from the engine. */
function useChain() {
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const { one } = useLabDerived();
  const x = data.X[0][0];
  const y = data.Y[0][0];
  const w1 = net.layers[0].W[0][0];
  const b1 = net.layers[0].b[0];
  const w2 = net.layers[1].W[0][0];
  const b2 = net.layers[1].b[0];
  const z1 = one.trace.z[0][0];
  const a1 = one.trace.a[1][0];
  const z2 = one.trace.z[1][0];
  const out = one.trace.a[2][0];
  const r1 = actDeriv('relu', z1);
  const r2 = actDeriv('relu', z2);
  const dL = one.dOut[0];
  const dout_da1 = r2 * w2;
  const da1_dw1 = r1 * x;
  return {
    net,
    x,
    y,
    w1,
    b1,
    w2,
    b2,
    z1,
    a1,
    z2,
    out,
    r1,
    r2,
    L: one.loss,
    dL,
    dout_da1,
    da1_dw1,
    gw1: one.back.grads.W[0][0][0],
    gw2: one.back.grads.W[1][0][0],
    gb1: one.back.grads.b[0][0],
    gb2: one.back.grads.b[1][0],
  };
}

const lossOf = (net: Network, x: number, y: number) => lossSample('mse', predict(net, [x]), [y]);

function Explain() {
  const { tr } = useT();
  return (
    <>
      <Cycle active={1} />
      <Reveal at={1}>
        <P>
          {bi(
            'Two neurons in a row. The input $\\cx{x}$ is multiplied by $\\cw{w_1}$, passed through ReLU to give $\\ca{a_1}$, then multiplied by $\\cw{w_2}$ and passed through ReLU again to give the output $\\cyhat{\\hat y}$. The loss compares it with the target $\\cy{y}$.',
            'دو نورون پشت سر هم. ورودی $\\cx{x}$ در $\\cw{w_1}$ ضرب می‌شود، از ReLU رد می‌شود و $\\ca{a_1}$ را می‌سازد، بعد در $\\cw{w_2}$ ضرب می‌شود و دوباره از ReLU رد می‌شود تا خروجی $\\cyhat{\\hat y}$ ساخته شود. خطا آن را با هدف $\\cy{y}$ مقایسه می‌کند.',
          )}
        </P>
        <Tex display>{'\\cw{w_1} \\;\\to\\; \\ca{a_1} \\;\\to\\; \\cyhat{\\hat y} \\;\\to\\; \\cL{L}'}</Tex>
      </Reveal>
      <Reveal at={2}>
        <P>
          {bi(
            '$\\cw{w_1}$ and $\\cw{w_2}$ are independent knobs: changing $w_1$ does not change $w_2$. But $w_1$ has no direct wire to the loss. It can only affect the loss by first changing $a_1$, which changes $\\hat y$, which changes $L$.',
            '$\\cw{w_1}$ و $\\cw{w_2}$ دو پیچ مستقل‌اند: تغییر $w_1$ چیزی از $w_2$ را عوض نمی‌کند. اما $w_1$ هیچ سیم مستقیمی به خطا ندارد. فقط می‌تواند اول $a_1$ را عوض کند، که $\\hat y$ را عوض می‌کند، که $L$ را عوض می‌کند.',
          )}
        </P>
      </Reveal>
      <Reveal at={3}>
        <Predict
          id={ID + '-dir'}
          question={bi('At the start ($\\hat y = 12$, $y = 10$), if $w_1$ increases a little, does the loss go up or down?', 'در شروع ($\\hat y = 12$ و $y = 10$)، اگر $w_1$ کمی زیاد شود، خطا بالا می‌رود یا پایین؟')}
          choices={[
            { id: 'up', label: bi('Up', 'بالا') },
            { id: 'down', label: bi('Down', 'پایین') },
            { id: 'same', label: bi('No change', 'تغییری نمی‌کند') },
          ]}
          answer="up"
          explain={bi(
            'The output is already too big. A larger $w_1$ makes $a_1$ bigger, so $\\hat y$ grows further from 10. That is why $dL/dw_1 = +16$ is positive. Try the nudge test on the right.',
            'خروجی از قبل بزرگ‌تر از هدف است. $w_1$ بزرگ‌تر یعنی $a_1$ بزرگ‌تر، پس $\\hat y$ از ۱۰ دورتر می‌شود. برای همین $dL/dw_1 = +16$ مثبت است. آزمون تکان کوچک را در سمت راست امتحان کن.',
          )}
        />
      </Reveal>
      <Reveal at={4}>
        <h3>{tr(bi('Sensitivities multiply', 'حساسیت‌ها در هم ضرب می‌شوند'))}</h3>
        <P>
          {bi(
            'Each arrow has a local derivative: how much the next quantity moves per unit change of the previous one. A small change $\\Delta w_1$ is multiplied by each local derivative in turn as it travels to the loss:',
            'هر فلش یک مشتق محلی دارد: اینکه مقدار بعدی به ازای یک واحد تغییر در مقدار قبلی چقدر جابه‌جا می‌شود. یک تغییر کوچک $\\Delta w_1$ در مسیرش تا خطا، به ترتیب در هر مشتق محلی ضرب می‌شود:',
          )}
        </P>
        <Tex display>{'\\cg{\\frac{dL}{dw_1}} = \\frac{dL}{d\\hat y}\\cdot\\frac{d\\hat y}{da_1}\\cdot\\frac{da_1}{dw_1}'}</Tex>
        <P>
          {bi(
            'Multiplying is natural: if $a_1$ moves 2 units per unit of $w_1$, and $\\hat y$ moves 2 units per unit of $a_1$, then $\\hat y$ moves $2\\times2 = 4$ units per unit of $w_1$.',
            'ضرب کردن طبیعی است: اگر $a_1$ به ازای هر واحد $w_1$ دو واحد جابه‌جا شود، و $\\hat y$ به ازای هر واحد $a_1$ دو واحد، پس $\\hat y$ به ازای هر واحد $w_1$ برابر $2\\times2 = 4$ واحد جابه‌جا می‌شود.',
          )}
        </P>
      </Reveal>
      <Reveal at={5}>
        <Callout>
          {tr(
            bi(
              'Drag w1 below 0 on the right. The first ReLU outputs 0 and its slope becomes 0, so the whole chain becomes 0: the loss no longer reacts to small changes of w1 (or w2).',
              'w1 را در سمت راست به زیر صفر ببر. ReLU اول صفر می‌دهد و شیبش صفر می‌شود، پس کل زنجیره صفر می‌شود: خطا دیگر به تغییرهای کوچک w1 (یا w2) واکنشی نشان نمی‌دهد.',
            ),
          )}
        </Callout>
        <Advanced title={bi('Bias gradients and the ReLU convention', 'گرادیان بایاس‌ها و قرارداد ReLU')}>
          <Tex display>{"\\frac{\\partial L}{\\partial b_2} = \\frac{dL}{d\\hat y}\\cdot \\relu'(z_2)\\cdot 1, \\qquad \\frac{\\partial L}{\\partial b_1} = \\frac{dL}{d\\hat y}\\cdot \\relu'(z_2)\\,w_2\\cdot \\relu'(z_1)\\cdot 1"}</Tex>
          <P>
            {bi(
              'A bias enters $z$ with coefficient 1, so its local derivative is 1 where a weight\'s is its input. At the starting point: $\\partial L/\\partial b_2 = 4$ and $\\partial L/\\partial b_1 = 4\\cdot2 = 8$.',
              'بایاس با ضریب ۱ وارد $z$ می‌شود، پس مشتق محلی‌اش ۱ است، در حالی که برای وزن برابر ورودی آن است. در نقطه شروع: $\\partial L/\\partial b_2 = 4$ و $\\partial L/\\partial b_1 = 4\\cdot2 = 8$.',
            )}
          </P>
          <P>
            {bi(
              'ReLU is not differentiable at $z = 0$. This studio uses $\\relu\'(0) = 0$ (as PyTorch does), so when $z_1$ is exactly 0 the gradient through it is 0.',
              'ReLU در $z = 0$ مشتق‌پذیر نیست. این استودیو مثل PyTorch از $\\relu\'(0) = 0$ استفاده می‌کند، پس وقتی $z_1$ دقیقاً صفر است گرادیانی از آن رد نمی‌شود.',
            )}
          </P>
        </Advanced>
      </Reveal>
    </>
  );
}

function FactorCards() {
  const { tr } = useT();
  const [s, set] = useS();
  const c = useChain();
  const reduced = useUi((u) => u.reducedMotion);
  const isW1 = s.target === 'w1';
  const cards: { id: Factor; label: string; v: number; desc: string }[] = isW1
    ? [
        { id: 'dL', label: '\\frac{dL}{d\\hat y}', v: c.dL, desc: tr(bi('loss per unit of output', 'خطا به ازای هر واحد خروجی')) },
        { id: 'dout', label: '\\frac{d\\hat y}{da_1}', v: c.dout_da1, desc: tr(bi('output per unit of a₁', 'خروجی به ازای هر واحد a₁')) },
        { id: 'da1', label: '\\frac{da_1}{dw_1}', v: c.da1_dw1, desc: tr(bi('a₁ per unit of w₁', 'a₁ به ازای هر واحد w₁')) },
      ]
    : [
        { id: 'dL', label: '\\frac{dL}{d\\hat y}', v: c.dL, desc: tr(bi('loss per unit of output', 'خطا به ازای هر واحد خروجی')) },
        { id: 'dout', label: '\\frac{d\\hat y}{dw_2}', v: c.r2 * c.a1, desc: tr(bi('output per unit of w₂', 'خروجی به ازای هر واحد w₂')) },
      ];
  const shown = Math.min(s.composed, cards.length);
  const partial = cards.slice(0, shown).reduce((p, k) => p * k.v, 1);
  const total = isW1 ? c.gw1 : c.gw2;
  return (
    <div className="stack">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <Segmented
          label="target"
          value={s.target}
          onChange={(target) => set({ target, focus: 'dL', composed: target === 'w1' ? 3 : 2 })}
          options={[
            { value: 'w1', label: 'dL/dw₁' },
            { value: 'w2', label: 'dL/dw₂' },
          ]}
        />
        <div className="row">
          <button className="btn sm" onClick={() => set({ composed: 0 })} data-testid="compose-reset">
            ↺
          </button>
          <button className="btn sm primary" onClick={() => set({ composed: Math.min(cards.length, s.composed + 1) })} disabled={shown >= cards.length} data-testid="compose-next">
            × {tr(bi('multiply next', 'ضرب بعدی'))}
          </button>
        </div>
      </div>
      <div className="factor-row" data-testid="factor-row">
        {cards.map((k, i) => (
          <div key={k.id} style={{ display: 'contents' }}>
            {i > 0 && <span className="factor-op">×</span>}
            <button
              type="button"
              className={`factor ${s.focus === k.id ? 'on' : ''}`}
              style={{ opacity: i < shown ? 1 : 0.35, transition: reduced ? undefined : 'opacity .35s' }}
              onClick={() => set({ focus: k.id })}
              data-testid={`factor-${k.id}`}
            >
              <Tex>{k.label}</Tex>
              <div className="fv c-g">{fmt(k.v, 4)}</div>
              <div className="small muted">{k.desc}</div>
            </button>
          </div>
        ))}
        <span className="factor-op">=</span>
        <div className="factor on" style={{ cursor: 'default' }}>
          <Tex>{isW1 ? '\\frac{dL}{dw_1}' : '\\frac{dL}{dw_2}'}</Tex>
          <div className="fv c-g" data-testid="chain-total">
            {shown === cards.length ? fmt(total, 4) : `${fmt(partial, 4)}…`}
          </div>
          <div className="small muted">{tr(bi('backprop (engine)', 'Backprop (موتور)'))}: {fmt(total, 4)}</div>
        </div>
      </div>
    </div>
  );
}

function Center() {
  return (
    <div className="center-stack">
      <div className="canvas-wrap" style={{ minHeight: 240 }}>
        <NetworkCanvas />
      </div>
      <FactorCards />
    </div>
  );
}

function FactorDetail() {
  const { tr } = useT();
  const [s] = useS();
  const c = useChain();
  const d = s.delta;
  // Real perturbation through the engine, compared with the linear (derivative) prediction.
  const nudged = useMemo(() => {
    const p = s.target === 'w1' ? { kind: 'W' as const, layer: 0, j: 0, i: 0, id: 'W1[0,0]' } : { kind: 'W' as const, layer: 1, j: 0, i: 0, id: 'W2[0,0]' };
    const n2 = setParam(c.net, p, (s.target === 'w1' ? c.w1 : c.w2) + d);
    const z1 = n2.layers[0].W[0][0] * c.x + n2.layers[0].b[0];
    const a1 = Math.max(0, z1);
    const out = predict(n2, [c.x])[0];
    return { a1, out, L: lossOf(n2, c.x, c.y) };
  }, [c, d, s.target]);
  if (s.target === 'w2') {
    return (
      <div className="stack">
        <Tex display>{`\\cg{\\frac{dL}{dw_2}} = \\frac{dL}{d\\hat y}\\cdot\\relu'(z_2)\\cdot \\ca{a_1} = ${texP(c.dL)}\\cdot${tex(c.r2)}\\cdot${texP(c.a1)} = ${tex(c.gw2, 4)}`}</Tex>
        <NudgeTable rows={[[tr(bi('output ŷ', 'خروجی ŷ')), nudged.out - c.out, c.r2 * c.a1 * d], ['L', nudged.L - c.L, c.gw2 * d]]} />
      </div>
    );
  }
  const info: Record<Factor, { what: string; depends: string; formula: string; value: number; actual: number; predicted: number }> = {
    da1: {
      what: tr(bi('w₁ changes', 'w₁ تغییر می‌کند')),
      depends: tr(bi('a₁ = ReLU(w₁·x + b₁) depends on it', 'a₁ = ReLU(w₁·x + b₁) به آن وابسته است')),
      formula: `\\frac{da_1}{dw_1} = \\relu'(z_1)\\cdot\\cx{x} = ${tex(c.r1)}\\cdot${texP(c.x)} = ${tex(c.da1_dw1)}`,
      value: c.da1_dw1,
      actual: nudged.a1 - c.a1,
      predicted: c.da1_dw1 * d,
    },
    dout: {
      what: tr(bi('a₁ changes', 'a₁ تغییر می‌کند')),
      depends: tr(bi('ŷ = ReLU(w₂·a₁ + b₂) depends on it', 'ŷ = ReLU(w₂·a₁ + b₂) به آن وابسته است')),
      formula: `\\frac{d\\hat y}{da_1} = \\relu'(z_2)\\cdot\\cw{w_2} = ${tex(c.r2)}\\cdot${texP(c.w2)} = ${tex(c.dout_da1)}`,
      value: c.dout_da1,
      actual: nudged.out - c.out,
      predicted: c.dout_da1 * c.da1_dw1 * d,
    },
    dL: {
      what: tr(bi('ŷ changes', 'ŷ تغییر می‌کند')),
      depends: tr(bi('L = (ŷ − y)² depends on it', 'L = (ŷ − y)² به آن وابسته است')),
      formula: `\\frac{dL}{d\\hat y} = 2(\\cyhat{\\hat y}-\\cy{y}) = 2(${tex(c.out)} - ${texP(c.y)}) = ${tex(c.dL)}`,
      value: c.dL,
      actual: nudged.L - c.L,
      predicted: c.gw1 * d,
    },
  };
  const f = info[s.focus];
  return (
    <div className="stack" data-testid="factor-detail">
      <div className="small">
        <strong>{tr(bi('What changes', 'چه چیزی تغییر می‌کند'))}:</strong> {f.what}. <strong>{tr(bi('What depends on it', 'چه چیزی به آن وابسته است'))}:</strong> {f.depends}.
      </div>
      <Tex display>{f.formula}</Tex>
      <NudgeTable
        rows={[
          ['a₁', nudged.a1 - c.a1, c.da1_dw1 * d],
          ['ŷ', nudged.out - c.out, c.dout_da1 * c.da1_dw1 * d],
          ['L', nudged.L - c.L, c.gw1 * d],
        ]}
      />
      <P className="small muted">
        {bi(
          'Predicted = product of the local derivatives so far × Δw₁. For small Δ the real change and the prediction agree; for large Δ they drift apart (the derivative is local).',
          'پیش‌بینی = حاصل‌ضرب مشتق‌های محلی تا آنجا × Δw₁. برای Δ کوچک تغییر واقعی و پیش‌بینی با هم می‌خوانند؛ برای Δ بزرگ از هم فاصله می‌گیرند (مشتق محلی است).',
        )}
      </P>
    </div>
  );
}

function NudgeTable({ rows }: { rows: [string, number, number][] }) {
  const { tr } = useT();
  const [s] = useS();
  return (
    <table className="tbl" data-testid="nudge-table">
      <thead>
        <tr>
          <th>Δw = {fmt(s.delta, 4)}</th>
          <th className="num">{tr(bi('real change', 'تغییر واقعی'))}</th>
          <th className="num">{tr(bi('derivative × Δ', 'مشتق × Δ'))}</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([k, a, p]) => (
          <tr key={k}>
            <td>{k}</td>
            <td className="num">{fmtSigned(a, 5)}</td>
            <td className="num c-g">{fmtSigned(p, 5)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Right() {
  const { tr } = useT();
  const [s, set] = useS();
  const c = useChain();
  const setParamStore = useLab((st) => st.setParam);
  const setInput = useLab((st) => st.setInput);
  const setTarget = useLab((st) => st.setTarget);
  return (
    <div className="stack">
      <Slider label="x" value={c.x} min={-3} max={3} step={0.1} color="x" onChange={(v) => setInput(0, 0, v)} testId="chain-x" />
      <Slider label="w₁" value={c.w1} min={-2} max={5} step={0.05} color="w" onChange={(v) => setParamStore({ kind: 'W', layer: 0, j: 0, i: 0 }, v)} testId="chain-w1" />
      <Slider label="w₂" value={c.w2} min={-2} max={5} step={0.05} color="w" onChange={(v) => setParamStore({ kind: 'W', layer: 1, j: 0, i: 0 }, v)} testId="chain-w2" />
      <Slider label="b₁" value={c.b1} min={-5} max={5} step={0.1} color="b" onChange={(v) => setParamStore({ kind: 'b', layer: 0, j: 0, i: -1 }, v)} />
      <Slider label="b₂" value={c.b2} min={-5} max={5} step={0.1} color="b" onChange={(v) => setParamStore({ kind: 'b', layer: 1, j: 0, i: -1 }, v)} />
      <Slider label="y" value={c.y} min={-5} max={20} step={0.5} color="y" onChange={(v) => setTarget(0, 0, v)} />
      <h4>{tr(bi('Forward', 'رو به جلو'))}</h4>
      <Tex display>{`\\cz{z_1} = \\cw{w_1}\\cx{x} + \\cb{b_1} = ${texP(c.w1)}\\cdot${texP(c.x)} + ${texP(c.b1)} = ${tex(c.z1)}`}</Tex>
      <Tex display>{`\\ca{a_1} = \\relu(${tex(c.z1)}) = ${tex(c.a1)}`}</Tex>
      <Tex display>{`\\cz{z_2} = \\cw{w_2}\\ca{a_1} + \\cb{b_2} = ${texP(c.w2)}\\cdot${texP(c.a1)} + ${texP(c.b2)} = ${tex(c.z2)}`}</Tex>
      <Tex display>{`\\cyhat{\\hat y} = \\relu(${tex(c.z2)}) = ${tex(c.out)}`}</Tex>
      <Tex display>{`\\cL{L} = (\\cyhat{\\hat y}-\\cy{y})^2 = (${tex(c.out)} - ${texP(c.y)})^2 = ${tex(c.L)}`}</Tex>
      <h4>{tr(bi('Backward', 'رو به عقب'))}</h4>
      <Tex display>{`\\cg{\\frac{dL}{dw_1}} = ${texP(c.dL)}\\times${texP(c.dout_da1)}\\times${texP(c.da1_dw1)} = ${tex(c.gw1, 4)}`}</Tex>
      <Tex display>{`\\cg{\\frac{dL}{dw_2}} = ${texP(c.dL)}\\times${tex(c.r2)}\\times${texP(c.a1)} = ${tex(c.gw2, 4)}`}</Tex>
      <div className="stat-grid">
        <Stat k="∂L/∂b₁" v={fmt(c.gb1, 4)} color="g" />
        <Stat k="∂L/∂b₂" v={fmt(c.gb2, 4)} color="g" />
        <Stat k="ReLU′(z₁)" v={fmt(c.r1)} />
        <Stat k="ReLU′(z₂)" v={fmt(c.r2)} />
      </div>
      {(c.r1 === 0 || c.r2 === 0) && (
        <Callout tone="warn">
          {tr(bi('A ReLU is in its zero region (z ≤ 0): its slope is 0, so no gradient flows back through it.', 'یک ReLU در ناحیه صفر است (z ≤ 0): شیبش ۰ است، پس هیچ گرادیانی از آن به عقب رد نمی‌شود.'))}
        </Callout>
      )}
      <h4>{tr(bi('Selected factor', 'ضریب انتخاب‌شده'))}</h4>
      <Slider label="Δw" value={s.delta} min={0.001} max={1} step={0.001} onChange={(delta) => set({ delta })} digits={4} testId="chain-delta" />
      <FactorDetail />
    </div>
  );
}

function Bottom() {
  const { tr } = useT();
  const c = useChain();
  const [s] = useS();
  const isW1 = s.target === 'w1';
  const v = isW1 ? c.w1 : c.w2;
  const fL = (w: number) => {
    const n2 = setParam(c.net, isW1 ? { kind: 'W', layer: 0, j: 0, i: 0, id: '' } : { kind: 'W', layer: 1, j: 0, i: 0, id: '' }, w);
    return lossOf(n2, c.x, c.y);
  };
  const fOut = (w: number) => {
    const n2 = setParam(c.net, isW1 ? { kind: 'W', layer: 0, j: 0, i: 0, id: '' } : { kind: 'W', layer: 1, j: 0, i: 0, id: '' }, w);
    return predict(n2, [c.x])[0];
  };
  const g = isW1 ? c.gw1 : c.gw2;
  const dOut = isW1 ? c.dout_da1 * c.da1_dw1 : c.r2 * c.a1;
  const name = isW1 ? 'w₁' : 'w₂';
  return (
    <div className="viz-grid">
      <div>
        <div className="small muted">{tr(bi(`Output ŷ as a function of ${name} (others fixed)`, `خروجی ŷ بر حسب ${name} (بقیه ثابت)`))}</div>
        <Plot2D
          xDomain={[-2, 5]}
          yDomain="auto"
          height={190}
          xLabel={name}
          yLabel="ŷ"
          layers={[
            { kind: 'fn', f: fOut, color: 'yhat', width: 2.4 },
            { kind: 'hline', at: c.y, color: 'y', label: 'y' },
            { kind: 'line', p: [v, c.out], slope: dOut, color: 'g', dash: '6 4' },
            { kind: 'points', color: 'yhat', pts: [{ x: v, y: c.out }] },
          ]}
        />
      </div>
      <div>
        <div className="small muted">{tr(bi(`Loss as a function of ${name} (others fixed)`, `خطا بر حسب ${name} (بقیه ثابت)`))}</div>
        <Plot2D
          testId="chain-loss-plot"
          xDomain={[-2, 5]}
          yDomain="auto"
          height={190}
          xLabel={name}
          yLabel="L"
          layers={[
            { kind: 'fn', f: fL, color: 'L', width: 2.4 },
            { kind: 'line', p: [v, c.L], slope: g, color: 'g', dash: '6 4' },
            { kind: 'points', color: 'L', pts: [{ x: v, y: c.L, label: `dL/d${name} = ${fmt(g, 3)}` }] },
          ]}
        />
      </div>
    </div>
  );
}

export const lesson10: LessonDef = {
  id: ID,
  title: bi('The Chain Rule', 'قاعده زنجیره‌ای'),
  short: bi('Chain rule', 'قاعده زنجیره‌ای'),
  objectives: [
    bi('Trace how w₁ reaches the loss only through a₁ and ŷ', 'دنبال کنی که w₁ فقط از راه a₁ و ŷ به خطا می‌رسد'),
    bi('Compute dL/dw₁ = dL/dŷ · dŷ/da₁ · da₁/dw₁ = 4 · 2 · 2 = 16', 'dL/dw₁ = dL/dŷ · dŷ/da₁ · da₁/dw₁ = 4 · 2 · 2 = 16 را حساب کنی'),
    bi('Check each factor with a real small perturbation', 'هر ضریب را با یک تکان کوچک واقعی بررسی کنی'),
    bi('See what happens when a ReLU enters its zero region', 'ببینی وقتی یک ReLU وارد ناحیه صفر می‌شود چه اتفاقی می‌افتد'),
  ],
  presetId: 'chain-two',
  revealSteps: 5,
  Explain,
  Center,
  Right,
  Bottom,
  checkpoint: {
    question: bi('Why do we multiply derivatives along a chain?', 'چرا مشتق‌ها را در طول زنجیره در هم ضرب می‌کنیم؟'),
    choices: [
      { id: 'a', label: bi('Because w₂ depends on w₁', 'چون w₂ به w₁ وابسته است') },
      { id: 'b', label: bi('Because each step scales the change it receives by its own local rate', 'چون هر مرحله تغییری را که دریافت می‌کند با نرخ محلی خودش بزرگ یا کوچک می‌کند') },
      { id: 'c', label: bi('It is just a convention', 'فقط یک قرارداد است') },
    ],
    answer: 'b',
    explain: bi(
      'w₁ and w₂ are independent. A change in w₁ is scaled by da₁/dw₁, then by dŷ/da₁, then by dL/dŷ, so the overall rate is the product.',
      'w₁ و w₂ مستقل‌اند. تغییر w₁ اول در da₁/dw₁، بعد در dŷ/da₁، بعد در dL/dŷ ضرب می‌شود، پس نرخ کل حاصل‌ضرب آنهاست.',
    ),
  },
};
