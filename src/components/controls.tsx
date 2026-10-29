import { useEffect, useState, type ReactNode } from 'react';
import { fmt } from '../engine/numeric';

export type MathColor = 'x' | 'w' | 'b' | 'z' | 'a' | 'yhat' | 'y' | 'L' | 'g' | 'lr';

/** Number box that only commits finite values (on blur or Enter). */
export function NumberField({
  value,
  onChange,
  step = 0.1,
  label,
  disabled,
  digits = 4,
}: {
  value: number;
  onChange: (v: number) => void;
  step?: number;
  label: string;
  disabled?: boolean;
  digits?: number;
}) {
  const [text, setText] = useState(fmt(value, digits).replace('−', '-'));
  useEffect(() => setText(fmt(value, digits).replace('−', '-')), [value, digits]);
  const commit = () => {
    const v = Number(text);
    if (Number.isFinite(v)) onChange(v);
    else setText(fmt(value, digits).replace('−', '-'));
  };
  return (
    <input
      className="field num"
      aria-label={label}
      type="number"
      step={step}
      value={text}
      disabled={disabled}
      onChange={(e) => {
        setText(e.target.value);
        const v = Number(e.target.value);
        if (e.target.value !== '' && Number.isFinite(v)) onChange(v);
      }}
      onBlur={commit}
      onKeyDown={(e) => e.key === 'Enter' && commit()}
    />
  );
}

export function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step = 0.01,
  color,
  frozen,
  frozenLabel,
  digits = 3,
  testId,
}: {
  label: ReactNode;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  color?: MathColor;
  frozen?: boolean;
  frozenLabel?: string;
  digits?: number;
  testId?: string;
}) {
  const ariaLabel = typeof label === 'string' ? label : testId ?? 'value';
  return (
    <div
      className={`slider ${frozen ? 'frozen' : ''}`}
      style={color ? ({ '--slider-color': `var(--c-${color})` } as React.CSSProperties) : undefined}
      data-testid={testId}
    >
      <label className={color ? `c-${color}` : undefined}>
        {label}
        {frozen && frozenLabel ? <span className="muted small"> · {frozenLabel}</span> : null}
      </label>
      <input
        type="range"
        dir="ltr"
        aria-label={ariaLabel}
        min={min}
        max={max}
        step={step}
        value={Math.min(max, Math.max(min, value))}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <NumberField value={value} onChange={onChange} step={step} label={ariaLabel} digits={digits} />
    </div>
  );
}

export function Segmented<T extends string | number>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: ReactNode }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.value)} type="button" aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Select<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <select className="select" aria-label={label} value={value} onChange={(e) => onChange(e.target.value as T)}>
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function Stat({ k, v, color, testId }: { k: ReactNode; v: ReactNode; color?: MathColor; testId?: string }) {
  return (
    <div className="stat">
      <div className="k">{k}</div>
      <div className={`v ${color ? `c-${color}` : ''}`} data-testid={testId}>
        {v}
      </div>
    </div>
  );
}
