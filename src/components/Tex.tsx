import katex from 'katex';
import { memo, useMemo } from 'react';

// Color macros so every equation uses the same semantic colors as the graphs and network:
//   \cx input, \cw weight, \cb bias, \cz pre-activation, \ca activation, \cyhat prediction,
//   \cy target, \cL loss, \cg gradient, \clr learning rate, \hl highlighted term.
const macros: Record<string, string> = {
  '\\cx': '\\htmlClass{c-x}{#1}',
  '\\cw': '\\htmlClass{c-w}{#1}',
  '\\cb': '\\htmlClass{c-b}{#1}',
  '\\cz': '\\htmlClass{c-z}{#1}',
  '\\ca': '\\htmlClass{c-a}{#1}',
  '\\cyhat': '\\htmlClass{c-yhat}{#1}',
  '\\cy': '\\htmlClass{c-y}{#1}',
  '\\cL': '\\htmlClass{c-L}{#1}',
  '\\cg': '\\htmlClass{c-g}{#1}',
  '\\clr': '\\htmlClass{c-lr}{#1}',
  '\\hl': '\\htmlClass{hl-box}{#1}',
  '\\relu': '\\operatorname{ReLU}',
};

const cache = new Map<string, string>();

export function renderTex(src: string, display: boolean): string {
  const key = (display ? 'D' : 'I') + src;
  let html = cache.get(key);
  if (!html) {
    html = katex.renderToString(src, {
      displayMode: display,
      throwOnError: false,
      trust: (ctx) => ctx.command === '\\htmlClass',
      strict: false,
      macros,
    });
    if (cache.size > 4000) cache.clear();
    cache.set(key, html);
  }
  return html;
}

/** KaTeX math. Always rendered left-to-right, including inside Persian text. */
export const Tex = memo(function Tex({
  children,
  display = false,
  className,
}: {
  children: string;
  display?: boolean;
  className?: string;
}) {
  const html = useMemo(() => renderTex(children, display), [children, display]);
  const Tag = display ? 'div' : 'span';
  return <Tag className={`ltr ${display ? 'eq' : ''} ${className ?? ''}`} dir="ltr" dangerouslySetInnerHTML={{ __html: html }} />;
});

/** Short alias for inline math inside prose. */
export const M = ({ children }: { children: string }) => <Tex>{children}</Tex>;
