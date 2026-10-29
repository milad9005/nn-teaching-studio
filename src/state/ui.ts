// UI preferences and navigation. Persisted to localStorage; never holds math.

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type Lang = 'en' | 'fa';
export type Theme = 'system' | 'light' | 'dark';
export type Mode = 'guided' | 'explore';
export type PanelId = 'left' | 'right' | 'bottom';

export interface UiState {
  lang: Lang;
  theme: Theme;
  mode: Mode;
  lessonIndex: number;
  presentation: boolean;
  /** Presentation step reveal: blocks with `at > reveal` stay hidden while presenting. */
  reveal: number;
  hideAdvanced: boolean;
  laser: boolean;
  draw: boolean;
  /** Bumped to clear annotations. */
  clearInk: number;
  reducedMotion: boolean;
  /** Training speed: optimizer steps per animation frame. */
  speed: number;
  collapsed: Record<PanelId, boolean>;
  maximized: null | 'center' | 'right';
  size: Record<PanelId, number>;
  completed: Record<string, boolean>;
  set: (p: Partial<UiState>) => void;
  toggle: (k: 'presentation' | 'hideAdvanced' | 'laser' | 'draw' | 'reducedMotion') => void;
  togglePanel: (p: PanelId) => void;
  goLesson: (i: number) => void;
}

const prefersReduced = () =>
  typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

export const useUi = create<UiState>()(
  persist(
    (set) => ({
      lang: 'en',
      theme: 'system',
      mode: 'guided',
      lessonIndex: 0,
      presentation: false,
      reveal: 0,
      hideAdvanced: false,
      laser: false,
      draw: false,
      clearInk: 0,
      reducedMotion: prefersReduced(),
      speed: 4,
      collapsed: { left: false, right: false, bottom: false },
      maximized: null,
      size: { left: 340, right: 380, bottom: 300 },
      completed: {},
      set: (p) => set(p),
      toggle: (k) => set((s) => ({ [k]: !s[k] }) as Partial<UiState>),
      togglePanel: (p) => set((s) => ({ collapsed: { ...s.collapsed, [p]: !s.collapsed[p] } })),
      goLesson: (i) => set({ lessonIndex: i, reveal: 0, mode: 'guided' }),
    }),
    {
      name: 'nnts-ui',
      partialize: (s) => ({
        lang: s.lang,
        theme: s.theme,
        mode: s.mode,
        lessonIndex: s.lessonIndex,
        hideAdvanced: s.hideAdvanced,
        speed: s.speed,
        collapsed: s.collapsed,
        size: s.size,
        completed: s.completed,
      }),
    },
  ),
);
