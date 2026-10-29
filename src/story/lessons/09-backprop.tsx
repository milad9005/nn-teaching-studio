// Lesson 9 · Backpropagation. One backward sweep, sharing pieces, gives every knob's derivative.
// The two-neuron chain (preset "chain-two") with blame travelling back from the loss; step 6 uses the
// "branching" preset (one knob, two paths); steps 7–8 show the training cycle and run it.
// Script: docs/storyboard.md, lesson 9.

import { useEffect, useLayoutEffect, useMemo, useState } from 'react';
import { bi, useT } from '../../i18n';
import { useLab } from '../../state/lab';
import { useUi } from '../../state/ui';
import { LossHistory } from '../../components/LabPanels';
import { Tex } from '../../components/Tex';
import { fmt, fmtSigned, tex, type PresetId } from '../../engine';
import type { StoryLesson } from '../types';
import { Appear, C, Formula, Stage, ValueCard } from '../kit';
import { ChainNet, NET_Y, POS, RAD, RateBadge, show, useChain, useClock, type Chip, type EdgeStyle, type NodeKey } from './08-chain';
import { backwardSweep, branchingOf, nudgeBranch, updateStep, type BlameAt } from './math-backprop';

const ID = 'backprop';
const BRANCH_STEP = 5;
const CYCLE_STEP = 6;
const TRAIN_STEP = 7;
/** Smaller steps than the preset's, so the loss falls slowly enough to watch. */
const TRAIN_LR = 0.002;
const TRAIN_ITERS = 30;
const TRAIN_EVERY_MS = 220;

const presetFor = (step: number): PresetId => (step === BRANCH_STEP ? 'branching' : 'chain-two');

/** Keeps the lab store in the right network for every step, whichever way the learner arrives. */
function useStepPreset(step: number) {
  useLayoutEffect(() => {
    const lab = useLab.getState();
    lab.setRunning(false);
    lab.select(null);
    lab.loadPreset(presetFor(step));
    lab.showAll();
  }, [step]);
}

// ---------------------------------------------------------------------------------------------
// Steps 1–5 and 7: the chain with blame flowing back.

const PULSE_MS = 3200;
const PULSE_GAP_MS = 900;
const REVEAL_MS = 5200;
const CYCLE_PHASE_MS = 2400;

/** A bracket over (or under) part of the network, naming a path. */
function Bracket({ x1, x2, y, down, label, color, dash }: { x1: number; x2: number; y: number; down: boolean; label: string; color: string; dash?: string }) {
  const tick = down ? 12 : -12;
  return (
    <g className="ch-pop">
      <path d={`M${x1},${y + tick} L${x1},${y} L${x2},${y} L${x2},${y + tick}`} fill="none" stroke={color} strokeWidth={4} strokeDasharray={dash} strokeLinejoin="round" />
      <text x={x1 + 6} y={down ? y - 10 : y + 28} fontSize={19} fontWeight={700} fill={color}>
        {label}
      </text>
    </g>
  );
}

/** The x position of something travelling from `from` to `to` (fraction p, eased). */
const along = (from: number, to: number, p: number) => from + (to - from) * (p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2);

const PHASES = [
  { key: 'forward', label: bi('Forward', 'جلو'), color: C.input, icon: '→' },
  { key: 'loss', label: bi('Loss', 'خطا'), color: C.loss, icon: '●' },
  { key: 'backward', label: bi('Backward', 'عقب'), color: C.slope, icon: '←' },
  {
    key: 'update',
    label: bi('Update', 'جابه‌جایی پیچ‌ها'),
    color: C.weight,
    icon: '⟳',
  },
] as const;

function CycleRow({ phase }: { phase: number }) {
  const { tr } = useT();
  return (
    <div
      dir="ltr"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        flexWrap: 'wrap',
        justifyContent: 'center',
      }}
    >
      {PHASES.map((p, i) => (
        <div key={p.key} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div
            style={{
              border: `3px solid ${p.color}`,
              borderRadius: 999,
              padding: '8px 20px',
              fontSize: '1.25em',
              fontWeight: 800,
              color: i === phase ? 'var(--surface)' : p.color,
              background: i === phase ? p.color : 'var(--surface)',
              transform: i === phase ? 'scale(1.08)' : 'none',
              transition: 'all 0.35s',
              opacity: i === phase ? 1 : 0.65,
            }}
          >
            <span className="num">{i + 1}</span> · {tr(p.label)} {p.icon}
          </div>
          {i < PHASES.length - 1 && <span style={{ fontSize: '1.6em', color: 'var(--line-strong)' }}>→</span>}
        </div>
      ))}
      <span
        style={{
          fontSize: '1.6em',
          color: phase === 3 ? C.weight : 'var(--line-strong)',
          transition: 'color 0.35s',
        }}
        title="repeat"
      >
        ↺
      </span>
    </div>
  );
}

function ChainScene({ step }: { step: number }) {
  const { tr } = useT();
  const c = useChain();
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const loss = useLab((s) => s.loss);
  const lossOpts = useLab((s) => s.lossOpts);
  const lr = useLab((s) => s.opt.lr);
  const sweep = useMemo(() => backwardSweep(net, data, loss, lossOpts), [net, data, loss, lossOpts]);
  const upd = useMemo(() => updateStep(net, data, lr, loss, lossOpts), [net, data, lr, loss, lossOpts]);
  const animated = step === 1 || step === 2 || step === 3 || step === CYCLE_STEP;
  const t = useClock(animated, step);
  // Step 2: the copies of "×4" slide out to both paths shortly after the step opens.
  const spread = step === 1 && t > 500;
  if (!c) return null;

  const blame = Object.fromEntries(sweep.map((b) => [b.at, b])) as Record<BlameAt, (typeof sweep)[number]>;
  const blameChip = (at: BlameAt, opacity = 1): Chip => {
    const b = blame[at];
    const sub = at === 'out' ? `2 × (${fmt(c.yhat)} − ${fmt(c.y)})` : `${fmt(b.from)} × ${fmt(b.rate)}`;
    return { top: fmt(b.value), sub, color: C.slope, opacity };
  };

  let chips: Partial<Record<NodeKey, Chip>> | undefined;
  let pulse: { x: number; color: string; label?: string } | null = null;
  let glow: Partial<Record<NodeKey, string>> | undefined;
  let edge: ((i: number) => EdgeStyle | undefined) | undefined;
  const fade: Partial<Record<NodeKey, number>> = { x: 0.4 };
  let phase = -1;

  if (step === 0 || step === 1) {
    edge = (i) => (i === 0 ? { opacity: 0.35 } : i === 4 && step === 1 ? { color: C.slope, width: 12 } : { color: C.weight, width: 6 });
    if (step === 1) glow = { out: C.slope, loss: C.slope };
  }

  if (step === 2 || step === 3) {
    // The blame pulse goes from the loss back to w₁.
    const from = POS.loss;
    const to = POS.w1;
    const period = step === 2 ? PULSE_MS + PULSE_GAP_MS : REVEAL_MS;
    const once = step === 3;
    const local = once ? Math.min(1, t / (REVEAL_MS - 600)) : (t % period) / PULSE_MS;
    const x = along(from, to, Math.min(1, local));
    const travelling = local < 1 && t > 0;
    if (travelling)
      pulse = {
        x,
        color: C.loss,
        label: step === 2 ? tr(bi('blame', 'تقصیر')) : undefined,
      };
    glow = {};
    (['out', 'w2', 'a1', 'w1'] as NodeKey[]).forEach((k) => {
      if (travelling && Math.abs(x - POS[k]) < RAD[k] + 12) glow![k] = C.loss;
    });
    edge = (i) =>
      i === 0
        ? { opacity: 0.35 }
        : {
            color: travelling && x < (i === 4 ? POS.loss : POS[(['x', 'w1', 'a1', 'w2', 'out'] as NodeKey[])[i + 1]]) ? C.loss : 'var(--line-strong)',
            width: 5,
          };
    if (step === 3) {
      // A number appears as soon as the pulse has passed its place: 4, 24, 8, 16.
      const passed = (k: NodeKey) => (!travelling ? t > 0 : x <= POS[k] + 2);
      chips = {};
      (['out', 'w2', 'a1', 'w1'] as BlameAt[]).forEach((k) => {
        if (passed(k)) chips![k] = blameChip(k);
      });
    }
  }

  if (step === 4) {
    chips = {
      out: blameChip('out', 0.3),
      w2: blameChip('w2'),
      a1: blameChip('a1', 0.3),
      w1: blameChip('w1'),
    };
    glow = { w1: C.slope, w2: C.slope };
    edge = (i) => (i === 0 ? { opacity: 0.35 } : undefined);
  }

  if (step === CYCLE_STEP) {
    phase = Math.floor(t / CYCLE_PHASE_MS) % 4;
    const local = (t % CYCLE_PHASE_MS) / (CYCLE_PHASE_MS * 0.8);
    const p = Math.min(1, local);
    fade.x = 1;
    if (phase === 0) {
      const x = along(POS.x, POS.loss, p);
      pulse = { x, color: C.input };
      edge = (i) => ({
        color: x > POS[(['x', 'w1', 'a1', 'w2', 'out'] as NodeKey[])[i]] ? C.input : 'var(--line-strong)',
        width: 5,
      });
    } else if (phase === 1) {
      glow = { loss: C.loss };
      chips = {
        loss: {
          top: `L = ${fmt(c.L)}`,
          sub: `(${fmt(c.yhat)} − ${fmt(c.y)})²`,
          color: C.loss,
        },
      };
    } else if (phase === 2) {
      const x = along(POS.loss, POS.w1, p);
      pulse = { x, color: C.loss };
      chips = {};
      if (x <= POS.w2 + 2) chips.w2 = blameChip('w2');
      if (x <= POS.w1 + 2) chips.w1 = blameChip('w1');
    } else {
      glow = { w1: C.weight, w2: C.weight };
      chips = {
        w1: {
          top: `${fmt(upd.w1)} → ${fmt(upd.w1New)}`,
          sub: `− ${fmt(lr)} × ${fmt(upd.g1)}`,
          color: C.weight,
        },
        w2: {
          top: `${fmt(upd.w2)} → ${fmt(upd.w2New)}`,
          sub: `− ${fmt(lr)} × ${fmt(upd.g2)}`,
          color: C.weight,
        },
      };
    }
  }

  const shared = { x: (POS.out + POS.loss) / 2 - 8 };
  return (
    <Stage>
      <ChainNet c={c} edge={edge} fade={fade} glow={glow} chips={chips} pulse={pulse} maxWidth={step === CYCLE_STEP ? 960 : 1060}>
        {(step === 0 || step === 1) && (
          <>
            <Bracket x1={POS.w1 - RAD.w1} x2={POS.loss + RAD.loss} y={100} down label={tr(bi('path of w₁', 'مسیر w₁'))} color={C.weight} />
            <Bracket x1={POS.w2 - RAD.w2} x2={POS.loss + RAD.loss} y={262} down={false} label={tr(bi('path of w₂', 'مسیر w₂'))} color={C.weight} dash="10 7" />
          </>
        )}
        {step === 1 && (
          <>
            <RateBadge x={shared.x} y={NET_Y} text={`×${fmt(c.rLast)}`} show />
            {[-60, 102].map((dy) => (
              <g
                key={dy}
                style={{
                  transform: `translateY(${spread ? dy : 0}px)`,
                  opacity: spread ? 1 : 0,
                  transition: 'transform 0.9s ease, opacity 0.5s',
                }}
              >
                <RateBadge x={shared.x} y={NET_Y} text={`×${fmt(c.rLast)}`} show />
              </g>
            ))}
          </>
        )}
      </ChainNet>
      {step === 4 && (
        <div
          className="ch-cards"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 18,
            flexWrap: 'wrap',
            justifyContent: 'center',
          }}
          dir="ltr"
        >
          <ValueCard label={<Tex>{'\\frac{\\partial L}{\\partial w_1}'}</Tex>} value={fmt(c.gw1)} color={C.slope} />
          <ValueCard label={<Tex>{'\\frac{\\partial L}{\\partial w_2}'}</Tex>} value={fmt(c.gw2)} color={C.slope} />
          <span style={{ fontSize: '2em', color: 'var(--muted)' }}>⇒</span>
          <ValueCard label={tr(bi('gradient', 'گرادیان'))} value={`(${fmt(c.gw1)}, ${fmt(c.gw2)})`} color={C.slope} big />
        </div>
      )}
      {step === CYCLE_STEP && <CycleRow phase={phase} />}
    </Stage>
  );
}

// ---------------------------------------------------------------------------------------------
// Step 6: one knob, two paths.

const BX = {
  x: [70, 180],
  P: [300, 95],
  Q: [300, 265],
  R: [540, 95],
  S: [540, 265],
  out: [760, 180],
  loss: [915, 180],
} as const;
type BKey = keyof typeof BX;

function BranchScene() {
  const { tr } = useT();
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const loss = useLab((s) => s.loss);
  const lossOpts = useLab((s) => s.lossOpts);
  const b = useMemo(() => branchingOf(net, data, loss, lossOpts), [net, data, loss, lossOpts]);
  const nudge = useMemo(() => nudgeBranch(net, data, 0.001, loss, lossOpts), [net, data, loss, lossOpts]);
  const t = useClock(true);
  const showPaths = t > 300;
  const showSum = t > 1400;

  const val: Record<BKey, number> = {
    x: b.x,
    P: b.h1[0],
    Q: b.h1[1],
    R: b.h2[0],
    S: b.h2[1],
    out: b.yhat,
    loss: b.L,
  };
  const col: Record<BKey, string> = {
    x: C.input,
    P: 'var(--ink-2)',
    Q: 'var(--ink-2)',
    R: 'var(--ink-2)',
    S: 'var(--ink-2)',
    out: C.output,
    loss: C.loss,
  };
  const r = (k: BKey) => (k === 'x' ? 38 : 40);
  const W = (l: number, j: number, i: number) => net.layers[l].W[j][i];
  // Edges: [from, to, weight value].
  const edges: [BKey, BKey, number | null][] = [
    ['x', 'P', W(0, 0, 0)],
    ['x', 'Q', W(0, 1, 0)],
    ['P', 'R', W(1, 0, 0)],
    ['P', 'S', W(1, 1, 0)],
    ['Q', 'R', W(1, 0, 1)],
    ['Q', 'S', W(1, 1, 1)],
    ['R', 'out', W(2, 0, 0)],
    ['S', 'out', W(2, 0, 1)],
    ['out', 'loss', null],
  ];
  const lf = (a: BKey, z: BKey) => ((a === 'P' && z === 'S') || (a === 'Q' && z === 'R') ? 0.3 : 0.5);
  const seg = (a: BKey, z: BKey) => {
    const [x1, y1] = BX[a];
    const [x2, y2] = BX[z];
    const d = Math.hypot(x2 - x1, y2 - y1);
    const ux = (x2 - x1) / d;
    const uy = (y2 - y1) / d;
    return {
      x1: x1 + ux * r(a),
      y1: y1 + uy * r(a),
      x2: x2 - ux * r(z),
      y2: y2 - uy * r(z),
      mx: (x1 + x2) / 2,
      my: (y1 + y2) / 2,
      // Weight label: part-way along the edge (early on the two crossing edges), just above the line.
      lx: x1 + (x2 - x1) * lf(a, z) + uy * 16,
      ly: y1 + (y2 - y1) * lf(a, z) - ux * 16,
    };
  };
  // The two paths from the knob (x → P) to the output: via R and via S.
  const hidden2: BKey[] = ['R', 'S'];
  const pathEdges = b.paths.map(
    (p) =>
      [
        ['x', 'P'],
        ['P', hidden2[p.via]],
        [hidden2[p.via], 'out'],
        ['out', 'loss'],
      ] as [BKey, BKey][],
  );
  const factors = b.paths.map((p) => [W(1, p.via, 0), W(2, 0, p.via), b.dOut]);
  const knob = seg('x', 'P');
  const label: Partial<Record<BKey, string>> = {
    x: tr(bi('input x', 'ورودی x')),
    out: tr(bi('output ŷ', 'خروجی ŷ')),
    loss: tr(bi('loss L', 'خطا L')),
  };

  return (
    <Stage>
      <svg viewBox="0 0 980 360" width="100%" style={{ maxWidth: 940, overflow: 'visible' }} role="img" aria-label="branching network" direction="ltr">
        <style>
          {
            '@keyframes ch-pop-k{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}} .ch-pop{animation:ch-pop-k .55s ease both} :root[data-motion=reduced] .ch-pop{animation:none}'
          }
        </style>
        {edges.map(([a, z, w]) => {
          const s = seg(a, z);
          return (
            <g key={`${a}${z}`}>
              <line x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2} stroke="var(--line-strong)" strokeWidth={3} />
              {w !== null && !(a === 'x' && z === 'P') && (
                <text
                  x={s.lx}
                  y={s.ly}
                  dominantBaseline="middle"
                  textAnchor="middle"
                  fontSize={17}
                  fontWeight={700}
                  fill={C.weight}
                  opacity={0.75}
                >
                  {fmt(w)}
                </text>
              )}
            </g>
          );
        })}
        {/* the two paths of the knob */}
        {showPaths &&
          pathEdges.map((pe, k) =>
            pe.map(([a, z]) => {
              const s = seg(a, z);
              const off = k === 0 ? -4 : 4;
              return (
                <line
                  key={`${k}${a}${z}`}
                  className="ch-pop"
                  x1={s.x1}
                  y1={s.y1 + off}
                  x2={s.x2}
                  y2={s.y2 + off}
                  stroke={C.slope}
                  strokeWidth={6}
                  strokeDasharray={k === 1 ? '12 8' : undefined}
                  strokeLinecap="round"
                />
              );
            }),
          )}
        {/* nodes */}
        {(Object.keys(BX) as BKey[]).map((k) => {
          const [x, y] = BX[k];
          return (
            <g key={k}>
              <circle cx={x} cy={y} r={r(k)} fill="var(--surface)" stroke={col[k]} strokeWidth={4} />
              <text x={x} y={y + 9} textAnchor="middle" fontSize={k === 'loss' ? 24 : 27} fontWeight={800} fill={col[k]}>
                {fmt(val[k], 2)}
              </text>
              {label[k] && (
                <text x={x} y={y + r(k) + 28} textAnchor="middle" fontSize={19} fontWeight={600} fill="var(--muted)">
                  {label[k]}
                </text>
              )}
            </g>
          );
        })}
        {/* the knob itself, sitting on the edge x → P */}
        <g>
          <circle cx={knob.mx} cy={knob.my} r={28} fill="color-mix(in srgb, var(--c-w) 14%, var(--surface))" stroke={C.weight} strokeWidth={3.5} />
          <text x={knob.mx} y={knob.my + 9} textAnchor="middle" fontSize={24} fontWeight={800} fill={C.weight}>
            {fmt(b.w)}
          </text>
          <text x={knob.mx - 40} y={knob.my - 26} textAnchor="middle" fontSize={22} fontWeight={800} fill={C.weight}>
            w
          </text>
        </g>
        {/* the right answer */}
        <rect x={BX.loss[0] - 42} y={60} width={84} height={40} rx={10} fill="color-mix(in srgb, var(--c-y) 14%, var(--surface))" stroke={C.target} strokeWidth={2.5} />
        <text x={BX.loss[0]} y={88} textAnchor="middle" fontSize={21} fontWeight={800} fill={C.target}>
          y = {fmt(b.y)}
        </text>
        <line x1={BX.loss[0]} y1={100} x2={BX.loss[0]} y2={BX.loss[1] - 40} stroke={C.target} strokeWidth={2} strokeDasharray="4 4" />
        {/* each path's share */}
        {showPaths && b.paths.map((p, k) => <RateBadge key={k} x={660} y={k === 0 ? 48 : 330} text={fmt(p.share)} note={undefined} show />)}
        {showPaths &&
          b.paths.map((_, k) => (
            <text key={`f${k}`} x={660} y={k === 0 ? 92 : 290} textAnchor="middle" fontSize={17} fontWeight={600} fill="var(--muted)">
              {factors[k].map((f) => fmt(f)).join(' × ')}
            </text>
          ))}
      </svg>
      <Appear show={showSum} keep>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 28,
            flexWrap: 'wrap',
            justifyContent: 'center',
          }}
        >
          <Formula size={1.9}>{`\\frac{\\partial L}{\\partial w} = ${b.paths.map((p) => `\\cg{${tex(p.share)}}`).join(' + ')} = \\cg{${tex(b.grad)}}`}</Formula>
          <ValueCard
            label={tr(bi('real nudge: w + 0.001', 'تلنگر واقعی: w + 0.001'))}
            value={fmtSigned(nudge.change, 4)}
            color={C.loss}
            note={<span dir="ltr">{`L: ${fmt(nudge.before, 3)} → ${fmt(nudge.after, 4)}`}</span>}
          />
        </div>
      </Appear>
    </Stage>
  );
}

// ---------------------------------------------------------------------------------------------
// Step 8: repeat the cycle: training.

function TrainScene() {
  const { tr } = useT();
  const c = useChain();
  const reduced = useUi((s) => s.reducedMotion);
  const iter = useLab((s) => s.history[s.history.length - 1]?.iter ?? 0);
  const [run, setRun] = useState(0);
  useEffect(() => {
    const lab = useLab.getState();
    if (run > 0) lab.loadPreset('chain-two');
    useLab.getState().setOpt({ lr: TRAIN_LR });
    if (reduced) {
      useLab.getState().trainSteps(TRAIN_ITERS);
      return;
    }
    let n = 0;
    let id = 0;
    const start = window.setTimeout(() => {
      id = window.setInterval(() => {
        if (n >= TRAIN_ITERS) return window.clearInterval(id);
        useLab.getState().trainSteps(1);
        n++;
      }, TRAIN_EVERY_MS);
    }, 700);
    return () => {
      window.clearTimeout(start);
      window.clearInterval(id);
    };
  }, [run, reduced]);
  if (!c) return null;
  return (
    <Stage>
      <ChainNet c={c} maxWidth={820} fade={{ x: 0.5 }} glow={{ w1: C.weight, w2: C.weight }} />
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 24,
          flexWrap: 'wrap',
          justifyContent: 'center',
          width: '100%',
        }}
      >
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 10,
          }}
        >
          <ValueCard
            label={tr(bi('loss L', 'خطا L'))}
            value={show(c.L, 3)}
            color={C.loss}
            big
            note={
              <>
                {tr(bi('cycles', 'دور'))}:{' '}
                <span className="num" dir="ltr">
                  {iter}
                </span>
              </>
            }
          />
          <button className="btn" onClick={() => setRun((k) => k + 1)} data-testid="train-again">
            ↺ {tr(bi('Train again', 'دوباره آموزش'))}
          </button>
        </div>
        <div style={{ width: 520, maxWidth: '100%' }}>
          <LossHistory height={210} />
        </div>
      </div>
    </Stage>
  );
}

function Scene({ step }: { step: number }) {
  useStepPreset(step);
  const presetId = useLab((s) => s.presetId);
  // Wait one frame until the store holds this step's network.
  if (presetId !== presetFor(step)) return null;
  if (step === BRANCH_STEP) return <BranchScene />;
  if (step === TRAIN_STEP) return <TrainScene />;
  return <ChainScene step={step} />;
}

export const backpropLesson: StoryLesson = {
  id: ID,
  presetId: 'chain-two',
  Scene,
  steps: [
    {
      say: bi('We also want the derivative of the loss with respect to w₂. Do we start over?', 'مشتق خطا نسبت به w₂ را هم می‌خواهیم. باید همه را از اول حساب کنیم؟'),
      more: bi(
        'The network has two knobs, so we need two derivatives. The path of w₁ goes through neuron 1, w₂ and the output to the loss. The path of w₂ is shorter: w₂ → output → loss. Look at the end of the two paths: they finish the same way.',
        'شبکه دو پیچ دارد، پس دو مشتق لازم داریم. مسیر w₁ از نورون ۱، w₂ و خروجی رد می‌شود تا به خطا برسد. مسیر w₂ کوتاه‌تر است: w₂ ← خروجی ← خطا. به انتهای این دو مسیر نگاه کن: هر دو یک‌جور تمام می‌شوند.',
      ),
    },
    {
      say: bi('No! The last link (×4) is shared by both. Compute it once, use it twice.', 'نه! حلقه آخر (۴) بین هر دو مشترک است. یک بار حساب کن، دو بار استفاده کن.'),
      more: bi(
        'Both paths end with the same link: output → loss, with rate ×4. There is no need to work it out twice: we compute it once and give the same number to both paths. In a big network many paths share their last links, so this saves a huge amount of work.',
        'هر دو مسیر با یک حلقه تمام می‌شوند: خروجی ← خطا، با نرخ ×۴. لازم نیست آن را دو بار حساب کنیم: یک بار حسابش می‌کنیم و همان عدد را به هر دو مسیر می‌دهیم. در یک شبکه بزرگ، مسیرهای زیادی حلقه‌های آخرشان مشترک است، پس این کار خیلی در وقت صرفه‌جویی می‌کند.',
      ),
    },
    {
      say: bi(
        'So we start at the end and walk backwards: each neuron multiplies the blame it receives by its own rate and passes it back.',
        'پس از آخر شروع می‌کنیم و عقب‌عقب می‌رویم: هر نورون سهمِ تقصیری را که به او رسیده در نرخ خودش ضرب می‌کند و به عقب می‌فرستد.',
      ),
      more: bi(
        '"Blame" is just a nickname for the derivative: how much the loss would change if this spot moved a tiny bit. The red pulse starts at the loss and walks back. At every stop it is multiplied by that spot\'s own rate, exactly like the gears of lesson 8, only walked in the other direction.',
        '«تقصیر» فقط یک اسم خودمانی برای مشتق است: اگر این نقطه کمی تکان بخورد، خطا چقدر عوض می‌شود. نبض قرمز از خطا شروع می‌کند و به عقب می‌رود. در هر ایستگاه در نرخ همان‌جا ضرب می‌شود؛ درست مثل چرخ‌دنده‌های درس ۸، فقط در جهت برعکس.',
      ),
    },
    {
      say: bi(
        "4 reaches the output. w₂'s share: 4 × 6 = 24. Passed to the first neuron: 4 × 2 = 8. w₁'s share: 8 × 2 = 16.",
        '۴ به خروجی می‌رسد. سهم w₂: ۴ × ۶ = ۲۴. به نورون اول: ۴ × ۲ = ۸. سهم w₁: ۸ × ۲ = ۱۶.',
      ),
      more: bi(
        'At the output the blame is 4. w₂ multiplies the 6 coming out of neuron 1, so its share is $4 \\times 6 = 24$. Going further back we cross w₂ = 2, so neuron 1 gets $4 \\times 2 = 8$. w₁ multiplies the input 2, so its share is $8 \\times 2 = 16$, the same 16 we found with the chain in lesson 8.',
        'در خروجی، تقصیر ۴ است. w₂ عدد ۶ را که از نورون ۱ می‌آید ضرب می‌کند، پس سهمش $4 \\times 6 = 24$ است. وقتی عقب‌تر می‌رویم از w₂ = 2 رد می‌شویم، پس به نورون ۱ می‌رسد $4 \\times 2 = 8$. w₁ ورودی ۲ را ضرب می‌کند، پس سهمش $8 \\times 2 = 16$ است؛ همان ۱۶ که در درس ۸ با زنجیره پیدا کردیم.',
      ),
    },
    {
      say: bi(
        'One trip backwards gives us the derivative of every knob. This method is called backpropagation.',
        'با یک بار رفتن به عقب، مشتق همه پیچ‌ها را داریم. اسم این روش «پس‌انتشار» است.',
      ),
      more: bi(
        'One walk from the loss back to the start gave both knobs their numbers: 16 for w₁ and 24 for w₂. Put together, $(16, 24)$ is the gradient of lesson 7. No matter how many knobs a network has, one backward walk is enough.',
        'یک بار رفتن از خطا تا ابتدای شبکه، عدد هر دو پیچ را داد: ۱۶ برای w₁ و ۲۴ برای w₂. کنار هم، $(16, 24)$ همان گرادیان درس ۷ است. شبکه هر چقدر هم پیچ داشته باشد، یک بار رفتن به عقب کافی است.',
      ),
      recall: {
        lesson: 'gradient',
        note: bi("the gradient: the list of every knob's derivative", 'گرادیان: فهرست مشتق همه پیچ‌ها'),
      },
    },
    {
      say: bi(
        "What if a knob reaches the output along two paths? We add up both paths' shares: 3 + 3 = 6.",
        'اگر یک پیچ از دو راه به خروجی برسد؟ سهم هر دو راه را جمع می‌کنیم. ۳ + ۳ = ۶.',
      ),
      more: bi(
        'Here the knob w feeds a neuron that sends its output to two neurons, so a nudge of w reaches the output along two roads. Along each road we multiply the rates as before: $1 \\times 1 \\times 3 = 3$ and $0.5 \\times 2 \\times 3 = 3$. The two effects happen together, so they add: 6. A real nudge of 0.001 changes the loss by about 0.006.',
        'این‌جا پیچ w به نورونی می‌رسد که خروجی‌اش را به دو نورون می‌دهد، پس تلنگرِ w از دو راه به خروجی می‌رسد. در هر راه مثل قبل نرخ‌ها را ضرب می‌کنیم: $1 \\times 1 \\times 3 = 3$ و $0.5 \\times 2 \\times 3 = 3$. این دو اثر با هم اتفاق می‌افتند، پس جمع می‌شوند: ۶. یک تلنگر واقعی ۰٫۰۰۱ خطا را حدود ۰٫۰۰۶ عوض می‌کند.',
      ),
    },
    {
      say: bi(
        'Now we have the whole cycle: go forward, compute the loss, go backward, move the knobs one step.',
        'حالا کل چرخه را داریم: جلو برو، خطا را حساب کن، به عقب برگرد، پیچ‌ها را یک قدم جابه‌جا کن.',
      ),
      more: bi(
        'Forward: the input flows through and gives 12. Loss: compare with 10, the loss is 4. Backward: the blame flows back and gives 16 and 24. Update: each knob takes a small step against its own slope: w₁ becomes $3 - 0.01 \\times 16 = 2.84$ and w₂ becomes $2 - 0.01 \\times 24 = 1.76$. Then the cycle starts again.',
        'جلو: ورودی از شبکه رد می‌شود و ۱۲ می‌دهد. خطا: با ۱۰ مقایسه می‌کنیم، خطا ۴ است. عقب: تقصیر به عقب برمی‌گردد و ۱۶ و ۲۴ را می‌دهد. جابه‌جایی: هر پیچ یک قدم کوچک خلاف شیب خودش برمی‌دارد: w₁ می‌شود $3 - 0.01 \\times 16 = 2.84$ و w₂ می‌شود $2 - 0.01 \\times 24 = 1.76$. بعد چرخه از نو شروع می‌شود.',
      ),
    },
    {
      say: bi('Repeat this cycle thousands of times: that is training a neural network.', 'این چرخه را هزاران بار تکرار کن: این یعنی آموزش شبکه عصبی.'),
      more: bi(
        'Each round of the cycle moves the knobs a little, and the loss gets smaller. Here we use small steps (0.002) so you can watch the loss fall round by round towards 0. Real networks do exactly this, with millions of knobs and thousands of rounds.',
        'هر دور از چرخه پیچ‌ها را کمی جابه‌جا می‌کند و خطا کوچک‌تر می‌شود. این‌جا قدم‌ها را کوچک گرفته‌ایم (۰٫۰۰۲) تا ببینی خطا دور به دور به سمت صفر پایین می‌آید. شبکه‌های واقعی دقیقاً همین کار را می‌کنند، با میلیون‌ها پیچ و هزاران دور.',
      ),
    },
  ],
};
