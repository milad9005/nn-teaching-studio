// Small numeric helpers shared by plots and lessons.

export const linspace = (a: number, b: number, n: number) =>
  Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Drop trailing zeros after a decimal point: 1.500 -> 1.5, 2.000 -> 2, 100 -> 100. */
const trimZeros = (s: string) => (s.includes('.') ? s.replace(/\.?0+$/, '') : s);

/** Below this, a displayed value is floating-point noise around 0 (e.g. a nudge test at a minimum). */
export const NOISE = 1e-12;

/** Format a number for display: fixed digits, no "-0", em dash for non-finite. */
export function fmt(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? '—' : v > 0 ? '∞' : '−∞';
  if (Math.abs(v) < NOISE) v = 0;
  if (v !== 0 && (Math.abs(v) >= 1e5 || Math.abs(v) < 10 ** -digits)) {
    return v.toExponential(2).replace('-', '−');
  }
  const s = trimZeros(v.toFixed(digits));
  return s === '-0' ? '0' : s.replace('-', '−');
}

/** Same as fmt but always shows the sign. */
export const fmtSigned = (v: number, digits = 3) => (v > 0 ? '+' : '') + fmt(v, digits);

/** Plain ASCII formatting for KaTeX strings. */
export function tex(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? '\\text{NaN}' : v > 0 ? '\\infty' : '-\\infty';
  if (Math.abs(v) < NOISE) v = 0;
  if (v !== 0 && (Math.abs(v) >= 1e5 || Math.abs(v) < 10 ** -digits)) {
    const [m, e] = v.toExponential(2).split('e');
    return `${m}\\times10^{${Number(e)}}`;
  }
  const s = trimZeros(v.toFixed(digits));
  return s === '-0' ? '0' : s;
}

/** Wrap negative numbers in parentheses for use inside a product. */
export const texP = (v: number, digits = 3) => (v < 0 ? `(${tex(v, digits)})` : tex(v, digits));

export function niceTicks(lo: number, hi: number, target = 6): number[] {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= target) ?? 10 * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}
