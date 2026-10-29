// Lesson prose: inline math ($…$), bold (**…**), and left-to-right arithmetic inside Persian text.

import type { ReactNode } from 'react';
import { useT, type Bi } from '../i18n';
import { Tex } from './Tex';

/** Bilingual paragraph/inline text. */
export function T({ children }: { children: Bi }) {
  const { tr } = useT();
  return <>{tr(children)}</>;
}

/**
 * Text with inline math: write `{ en: 'If $w > 0$ ...', fa: '...' }` and $...$ becomes KaTeX.
 * Keeps prose and math in one string per language.
 */
export function P({ children, className }: { children: Bi | string; className?: string }) {
  const { tr } = useT();
  return <p className={className}>{withMath(tr(children))}</p>;
}
export function withMath(s: string): ReactNode[] {
  return s.split(/(\$[^$]+\$|\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith('$') && part.endsWith('$') && part.length > 2 ? (
      <Tex key={i}>{part.slice(1, -1)}</Tex>
    ) : part.startsWith('**') && part.endsWith('**') && part.length > 4 ? (
      <strong key={i}>{part.slice(2, -2)}</strong>
    ) : (
      <span key={i}>{isolateNumbers(part)}</span>
    ),
  );
}

// A number with an optional sign, then any "op number" pairs: 6 − 10 = −4, 3 × 2, −16, ۴ × ۲ × ۲ = ۱۶.
const DIGITS = '[0-9۰-۹]+(?:[٫.,][0-9۰-۹]+)*';
const NUM = `\\(?[−\\-]?\\(?${DIGITS}\\)?`;
// A term is a number or a single variable name (b, w₁, ŷ, η), so "b = −8" stays together too.
const TERM = `(?:${NUM}|(?<![A-Za-z])[A-Za-zŷη][₀-₉]?(?![A-Za-z]))`;
const RUN = new RegExp(`(${TERM}(?:\\s*[−\\-+×÷=]\\s*${TERM})*\\s*[−\\-+×÷=]\\s*${NUM}|${NUM}(?:\\s*[−\\-+×÷=]\\s*${TERM})+|[−\\-]${DIGITS})`, 'g');

/**
 * Plain-text arithmetic inside a sentence ("6 − 10 = −4") is written left to right in both
 * languages; isolating it keeps a Persian sentence from reordering its parts or moving the minus.
 */
function isolateNumbers(s: string): ReactNode {
  const parts = s.split(RUN);
  if (parts.length === 1) return s;
  return parts.map((p, i) => (i % 2 === 1 ? <bdi key={i} dir="ltr" className="num-run">{p}</bdi> : p));
}
