// The shared visual vocabulary of the course. Using the same machine, knob, meter and cards in
// every lesson is what lets the learner recognise "this is the same thing as before".

import type { CSSProperties, ReactNode } from 'react';
import { Tex } from '../components/Tex';
import { fmt } from '../engine';
import { useTween } from './useTween';

/** Formats an animated value without scientific notation near zero. */
const show = (v: number, digits: number) => fmt(Math.abs(v) < 0.5 * 10 ** -digits ? 0 : Number(v.toFixed(digits)), digits);

/** Semantic colours, the same everywhere. */
export const C = {
  input: 'var(--c-x)',
  weight: 'var(--c-w)',
  bias: 'var(--c-b)',
  output: 'var(--c-yhat)',
  target: 'var(--c-y)',
  loss: 'var(--c-L)',
  slope: 'var(--c-g)',
  rate: 'var(--c-lr)',
} as const;

/** Fades and slides children in when `show` becomes true. Keeps layout stable when `keep`. */
export function Appear({ show, children, keep = false, style }: { show: boolean; children: ReactNode; keep?: boolean; style?: CSSProperties }) {
  if (!show && !keep) return null;
  return (
    <div className={`appear ${show ? 'on' : ''}`} style={style} aria-hidden={!show}>
      {children}
    </div>
  );
}

/** A big centred stage for a scene; children are laid out in a column. */
export function Stage({ children, row = false }: { children: ReactNode; row?: boolean }) {
  return <div className={`stage ${row ? 'stage-row' : ''}`}>{children}</div>;
}

/** A rotary knob. `value` is mapped onto a 270° sweep between min and max. */
export function Knob({
  label,
  value,
  min,
  max,
  color = C.weight,
  size = 86,
  digits = 2,
  ms = 600,
}: {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  color?: string;
  size?: number;
  digits?: number;
  /** Animation time when the value changes; 0 follows live values (dragging, sweeps). */
  ms?: number;
}) {
  const v = useTween(value, ms);
  const t = Math.max(0, Math.min(1, (v - min) / (max - min)));
  const ang = -135 + 270 * t;
  const r = size / 2 - 8;
  const cx = size / 2;
  const arc = (a0: number, a1: number) => {
    const p = (a: number) => [cx + r * Math.sin((a * Math.PI) / 180), cx - r * Math.cos((a * Math.PI) / 180)];
    const [x0, y0] = p(a0);
    const [x1, y1] = p(a1);
    return `M${x0},${y0} A${r},${r} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${x1},${y1}`;
  };
  return (
    <div className="knob" dir="ltr">
      <svg width={size} height={size} role="img" aria-label={`knob ${fmt(value, digits)}`}>
        <path d={arc(-135, 135)} stroke="var(--line-strong)" strokeWidth={6} fill="none" strokeLinecap="round" />
        <path d={arc(-135, ang)} stroke={color} strokeWidth={6} fill="none" strokeLinecap="round" />
        <circle cx={cx} cy={cx} r={r - 12} fill="var(--surface)" stroke="var(--line-strong)" />
        <line
          x1={cx}
          y1={cx}
          x2={cx + (r - 16) * Math.sin((ang * Math.PI) / 180)}
          y2={cx - (r - 16) * Math.cos((ang * Math.PI) / 180)}
          stroke={color}
          strokeWidth={4}
          strokeLinecap="round"
        />
      </svg>
      <div className="knob-label" style={{ color }}>
        {label}
      </div>
      <div className="knob-value num">{show(v, digits)}</div>
    </div>
  );
}

/** The loss gauge: a red needle that rises with the loss. `ms = 0` follows live values (training). */
export function Meter({ value, max, label, color = C.loss, digits = 3, ms = 600 }: { value: number; max: number; label: ReactNode; color?: string; digits?: number; ms?: number }) {
  const v = useTween(value, ms);
  const t = Math.max(0, Math.min(1, v / max));
  const ang = -90 + 180 * t;
  const W = 150;
  const r = 60;
  const cx = W / 2;
  const cy = 72;
  return (
    <div className="meter" dir="ltr">
      <svg width={W} height={84} role="img" aria-label={`meter ${fmt(value, digits)}`}>
        <path d={`M${cx - r},${cy} A${r},${r} 0 0 1 ${cx + r},${cy}`} stroke="var(--line-strong)" strokeWidth={10} fill="none" />
        <path
          d={`M${cx - r},${cy} A${r},${r} 0 0 1 ${cx + r * Math.sin((ang * Math.PI) / 180)},${cy - r * Math.cos((ang * Math.PI) / 180)}`}
          stroke={color}
          strokeWidth={10}
          fill="none"
        />
        <line x1={cx} y1={cy} x2={cx + (r - 14) * Math.sin((ang * Math.PI) / 180)} y2={cy - (r - 14) * Math.cos((ang * Math.PI) / 180)} stroke="var(--ink)" strokeWidth={3} strokeLinecap="round" />
        <circle cx={cx} cy={cy} r={5} fill="var(--ink)" />
      </svg>
      <div className="meter-label" style={{ color }}>
        {label}
      </div>
      <div className="meter-value num" style={{ color }}>
        {show(v, digits)}
      </div>
    </div>
  );
}

/** A labelled value card: used to keep output, loss, derivative and gradient visibly apart. */
export function ValueCard({ label, value, color, note, big = false }: { label: ReactNode; value: ReactNode; color: string; note?: ReactNode; big?: boolean }) {
  return (
    <div className={`vcard ${big ? 'big' : ''}`} style={{ borderColor: color }}>
      <div className="vcard-label" style={{ color }}>
        {label}
      </div>
      <div className="vcard-value num" style={{ color }}>
        {value}
      </div>
      {note && <div className="vcard-note">{note}</div>}
    </div>
  );
}

/**
 * The number machine from lesson 1: input on one side, output on the other, a rule and
 * optional knobs inside. Always drawn left-to-right (numbers flow left to right).
 */
export function Machine({
  input,
  output,
  rule,
  knobs,
  inputLabel,
  outputLabel,
  pulse,
  locked = false,
}: {
  input?: ReactNode;
  output?: ReactNode;
  /** TeX shown on the box. */
  rule?: string;
  knobs?: ReactNode;
  inputLabel?: ReactNode;
  outputLabel?: ReactNode;
  /** Change this value to replay the "processing" pulse. */
  pulse?: string | number;
  locked?: boolean;
}) {
  return (
    <div className="machine" dir="ltr">
      <div className="machine-io">
        <div className="machine-slot in" style={{ borderColor: C.input, color: C.input }}>
          <span key={`i${pulse}`} className="slot-val num">
            {input ?? ''}
          </span>
        </div>
        {inputLabel && <div className="machine-cap">{inputLabel}</div>}
      </div>
      <div className="machine-arrow">→</div>
      <div key={`b${pulse}`} className={`machine-box ${pulse !== undefined ? 'pulse' : ''}`}>
        {locked && <span className="machine-lock">🔒</span>}
        {rule && <Tex display>{rule}</Tex>}
        {knobs && <div className="machine-knobs">{knobs}</div>}
      </div>
      <div className="machine-arrow">→</div>
      <div className="machine-io">
        <div className="machine-slot out" style={{ borderColor: C.output, color: C.output }}>
          <span key={`o${pulse}`} className="slot-val num">
            {output ?? ''}
          </span>
        </div>
        {outputLabel && <div className="machine-cap">{outputLabel}</div>}
      </div>
    </div>
  );
}

/** A centred formula line in the scene. */
export function Formula({ children, size = 1.4 }: { children: string; size?: number }) {
  return (
    <div className="scene-formula" style={{ fontSize: `${size}em` }}>
      <Tex display>{children}</Tex>
    </div>
  );
}

/** Small caption text inside a scene (axis notes, legends). */
export function Note({ children }: { children: ReactNode }) {
  return <p className="scene-note">{children}</p>;
}
