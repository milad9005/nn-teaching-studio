// Per-lesson state shared by a lesson's panels (Explain, Center, Right, Bottom).
// Lives outside the components so switching language or collapsing a panel never resets it.

import { create } from 'zustand';
import { useCallback } from 'react';

interface LessonStore {
  data: Record<string, unknown>;
  set: (key: string, value: unknown) => void;
  reset: (prefix: string) => void;
}

export const useLessonStore = create<LessonStore>((set) => ({
  data: {},
  set: (key, value) => set((s) => ({ data: { ...s.data, [key]: value } })),
  reset: (prefix) =>
    set((s) => ({ data: Object.fromEntries(Object.entries(s.data).filter(([k]) => !k.startsWith(prefix + ':'))) })),
}));

type Patch<T> = Partial<T> | ((prev: T) => Partial<T>);

/**
 * `const [s, set] = useLessonState('l07', { a: 1 })`. `set` merges a partial object.
 * The key prefix (before any ":") is the lesson id; resetLesson(id) clears it.
 */
export function useLessonState<T extends object>(lessonId: string, init: T): [T, (p: Patch<T>) => void] {
  const key = `${lessonId}:state`;
  const stored = useLessonStore((s) => s.data[key]) as T | undefined;
  const value = stored ?? init;
  const setStore = useLessonStore((s) => s.set);
  const update = useCallback(
    (p: Patch<T>) => {
      const prev = (useLessonStore.getState().data[key] as T | undefined) ?? init;
      const patch = typeof p === 'function' ? p(prev) : p;
      setStore(key, { ...prev, ...patch });
    },
    // init is a literal at the call site; its identity changing must not reset anything.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [key, setStore],
  );
  return [value, update];
}

export const resetLesson = (id: string) => useLessonStore.getState().reset(id);
