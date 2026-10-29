// Thin Plotly wrapper used for 3-D surfaces and contour maps (rotate, zoom, pan built in).
// Plotly is loaded lazily so lessons without 3-D plots stay fast.

import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { Data, Layout } from 'plotly.js';

type PlotlyModule = typeof import('plotly.js-dist-min');
let plotlyPromise: Promise<PlotlyModule> | null = null;
const loadPlotly = () => (plotlyPromise ??= import('plotly.js-dist-min').then((m) => (m as unknown as { default: PlotlyModule }).default ?? m));

/** Resolve a CSS variable (e.g. "--c-g") to a concrete color for Plotly. */
export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
}

export function useThemeKey() {
  const [k, setK] = useState(0);
  useEffect(() => {
    const mo = new MutationObserver(() => setK((v) => v + 1));
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => mo.disconnect();
  }, []);
  return k;
}

export function PlotlyChart({
  data,
  layout,
  height = 360,
  testId,
}: {
  data: Data[];
  layout: Partial<Layout>;
  height?: number;
  testId?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const themeKey = useThemeKey();
  const [ready, setReady] = useState(false);

  useLayoutEffect(() => {
    let alive = true;
    const el = ref.current;
    loadPlotly().then((Plotly) => {
      if (!alive || !el) return;
      const ink = cssVar('--ink-2');
      const line = cssVar('--line');
      Plotly.react(
        el,
        data,
        {
          height,
          margin: { l: 50, r: 10, t: 10, b: 40 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { family: 'IBM Plex Sans, system-ui, sans-serif', color: ink, size: 12 },
          showlegend: false,
          ...layout,
          xaxis: { gridcolor: line, zerolinecolor: cssVar('--line-strong'), ...layout.xaxis },
          yaxis: { gridcolor: line, zerolinecolor: cssVar('--line-strong'), ...layout.yaxis },
        },
        { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'] },
      );
      setReady(true);
    });
    return () => {
      alive = false;
    };
  }, [data, layout, height, themeKey]);

  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) loadPlotly().then((P) => P.purge(el));
    };
  }, []);

  return (
    <div dir="ltr" data-testid={testId} style={{ minHeight: height, position: 'relative' }}>
      {!ready && <div className="muted small" style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center' }}>…</div>}
      <div ref={ref} />
    </div>
  );
}
