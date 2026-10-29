// Minimal i18n: a dictionary per language plus inline { en, fa } pairs for lesson prose.
// Switching language only changes text; no math state is touched.

import { useCallback } from 'react';
import { useUi, type Lang } from '../state/ui';
import { en, fa, type StringKey } from './strings';

export type Bi = { en: string; fa: string };
export type { StringKey };

const dict: Record<Lang, Record<StringKey, string>> = { en, fa };

export function useT() {
  const lang = useUi((s) => s.lang);
  const t = useCallback((k: StringKey) => dict[lang][k] ?? en[k], [lang]);
  const tr = useCallback((b: Bi | string) => (typeof b === 'string' ? b : b[lang]), [lang]);
  return { t, tr, lang, dir: lang === 'fa' ? ('rtl' as const) : ('ltr' as const) };
}

export const bi = (en: string, fa: string): Bi => ({ en, fa });
