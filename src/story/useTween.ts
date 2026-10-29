// Smoothly animates numbers towards their target when a step changes, so the visual moves
// instead of jumping. Respects "reduce motion".
import { useEffect, useRef, useState } from 'react';
import { useUi } from '../state/ui';

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

/** Animate one number. */
export function useTween(target: number, ms = 700): number {
  return useTweenArray([target], ms)[0];
}

/** Animate several numbers together (same duration and easing). */
export function useTweenArray(targets: number[], ms = 700): number[] {
  const reduced = useUi((s) => s.reducedMotion);
  const [value, setValue] = useState(targets);
  const from = useRef(targets);
  const current = useRef(targets);
  const key = targets.join('|');
  useEffect(() => {
    if (reduced || ms <= 0) {
      current.current = targets;
      setValue(targets);
      return;
    }
    from.current = current.current.length === targets.length ? current.current : targets;
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / ms);
      const k = ease(t);
      const next = targets.map((v, i) => from.current[i] + (v - from.current[i]) * k);
      current.current = next;
      setValue(next);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, reduced, ms]);
  return value;
}

/** Counts up from 0 to `n` while active (for "play" steps); returns the current integer. */
export function useTicker(active: boolean, n: number, everyMs = 600): number {
  const [i, setI] = useState(0);
  useEffect(() => {
    if (!active) return setI(0);
    setI(0);
    const id = setInterval(() => setI((k) => (k >= n ? k : k + 1)), everyMs);
    return () => clearInterval(id);
  }, [active, n, everyMs]);
  return i;
}
