// Reusable panels bound to the shared lab store: run controls, training stats, loss history,
// parameter list, gradient check, data editor and architecture editor.

import { useEffect, useMemo, useState } from 'react';
import { useLab, useLabDerived } from '../state/lab';
import { useUi } from '../state/ui';
import { useT } from '../i18n';
import { PRESET_TEXT } from '../i18n/presets';
import { NumberField, Segmented, Select, Slider, Stat } from './controls';
import { Plot2D, type PlotLayer } from './Plot2D';
import {
  ELEMENTWISE,
  LIMITS,
  PRESET_IDS,
  fmt,
  gradCheck,
  predict,
  paramRefs,
  getParam,
  getGrad,
  type ActivationName,
  type LossName,
  type OptimizerName,
  type PresetId,
  type GradCheckRow,
} from '../engine';

/** Training speeds in optimizer steps per frame; values below 1 slow training down for teaching. */
export const SPEEDS = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32];

/** Drives continuous training while `running` is true. Mount once. */
export function useTrainingLoop() {
  const running = useLab((s) => s.running);
  const speed = useUi((s) => s.speed);
  useEffect(() => {
    if (!running) return;
    let raf = 0;
    let acc = 0;
    const tick = () => {
      const s = useLab.getState();
      if (!s.running) return;
      acc += speed;
      const n = Math.floor(acc);
      acc -= n;
      if (n > 0) s.trainSteps(n);
      const after = useLab.getState();
      const last = after.history[after.history.length - 1];
      if (after.error || (last && last.loss < 1e-10) || (last && last.iter >= 20000)) {
        after.setRunning(false);
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running, speed]);
}

export function RunControls({ compact = false }: { compact?: boolean }) {
  const { t } = useT();
  const s = useLab();
  const speed = useUi((u) => u.speed);
  const setUi = useUi((u) => u.set);
  const L = s.net.layers.length;
  const label = (() => {
    const st = s.stepper;
    if (!st) return null;
    if (st.phase === 'update') return t('run.optimizer');
    if (st.phase === 'forward') return `${t('run.forward')} ${Math.min(st.k, L + 2)}/${L + 2}`;
    return `${t('run.backward')} ${st.k}/${L + 1}`;
  })();
  return (
    <div className="run-controls" data-testid="run-controls">
      <div className="row">
        <button className="btn" onClick={s.forwardStep} disabled={s.running} data-testid="btn-forward">
          → {t('run.forward')}
        </button>
        <button className="btn" onClick={s.backwardStep} disabled={s.running} data-testid="btn-backward">
          ← {t('run.backward')}
        </button>
        <button className="btn" onClick={() => s.optimizerStep()} disabled={s.running} data-testid="btn-optimizer">
          ⟳ {t('run.optimizer')}
        </button>
        {s.stepper && (
          <button className="btn ghost sm" onClick={s.showAll}>
            {t('run.showAll')}
          </button>
        )}
      </div>
      {!compact && (
        <div className="row">
          <button className={`btn ${s.running ? '' : 'primary'}`} onClick={() => s.setRunning(!s.running)} data-testid="btn-train">
            {s.running ? `❚❚ ${t('run.pause')}` : `▶ ${t('run.play')}`}
          </button>
          <button className="btn" onClick={s.resetTraining} data-testid="btn-reset">
            ↺ {t('run.reset')}
          </button>
          <label className="small muted row" style={{ gap: 6 }}>
            {t('run.speed')}
            <input
              type="range"
              min={0}
              max={SPEEDS.length - 1}
              value={Math.max(0, SPEEDS.findIndex((v) => v >= speed))}
              onChange={(e) => setUi({ speed: SPEEDS[Number(e.target.value)] })}
              aria-label={t('run.speed')}
            />
            <span className="num">≈{Math.round(speed * 60)}/s</span>
          </label>
        </div>
      )}
      {label && <div className="small stepper-label">{label}</div>}
      {s.error && <div className="callout err small">{t(`err.${s.error}`)}</div>}
    </div>
  );
}

export function TrainStats() {
  const { t } = useT();
  const s = useLab();
  const der = useLabDerived();
  const last = s.history[s.history.length - 1];
  return (
    <div className="stat-grid" data-testid="train-stats">
      <Stat k={t('stat.iteration')} v={s.train.schedule.iteration} testId="stat-iter" />
      <Stat k={t('stat.epoch')} v={s.train.schedule.epoch} />
      <Stat k={t('stat.loss')} v={fmt(last?.loss ?? der.full.loss, 5)} color="L" testId="stat-loss" />
      {last?.val !== undefined && <Stat k={t('stat.valLoss')} v={fmt(last.val, 5)} color="L" />}
      <Stat k={t('stat.lr')} v={fmt(s.opt.lr, 4)} color="lr" />
      <Stat k={t('stat.optimizer')} v={t(`opt.${s.opt.name}`)} />
      <Stat k={t('stat.batch')} v={`${der.nextIdx.length}/${s.data.X.length}`} />
    </div>
  );
}

export function OptimizerControls() {
  const { t } = useT();
  const opt = useLab((s) => s.opt);
  const setOpt = useLab((s) => s.setOpt);
  const batch = useLab((s) => s.batch);
  const setBatch = useLab((s) => s.setBatch);
  const N = useLab((s) => s.data.X.length);
  return (
    <div className="stack">
      <Segmented<OptimizerName>
        label={t('stat.optimizer')}
        value={opt.name}
        onChange={(name) => setOpt({ name })}
        options={(['sgd', 'momentum', 'adam'] as const).map((v) => ({ value: v, label: t(`opt.${v}`) }))}
      />
      <Slider label={t('stat.lr')} value={opt.lr} min={0.001} max={1} step={0.001} color="lr" onChange={(lr) => setOpt({ lr })} digits={4} testId="lr-slider" />
      {opt.name === 'momentum' && <Slider label="β (momentum)" value={opt.momentum} min={0} max={0.99} step={0.01} onChange={(momentum) => setOpt({ momentum })} />}
      {N > 1 && (
        <div className="row">
          <span className="small">{t('batch.size')}</span>
          <Segmented<number>
            label={t('batch.size')}
            value={batch.size <= 0 || batch.size >= N ? 0 : batch.size}
            onChange={(size) => setBatch({ size })}
            options={[1, 2, 4].filter((b) => b < N).map((b) => ({ value: b, label: String(b) })).concat([{ value: 0, label: t('batch.full') }])}
          />
          <label className="small row" style={{ gap: 4 }}>
            <input type="checkbox" checked={batch.shuffle} onChange={(e) => setBatch({ shuffle: e.target.checked })} />
            {t('batch.shuffle')}
          </label>
        </div>
      )}
      <label className="small row" style={{ gap: 6 }}>
        <input type="checkbox" checked={opt.clipNorm !== null} onChange={(e) => setOpt({ clipNorm: e.target.checked ? 5 : null })} />
        {t('opt.clip')} {opt.clipNorm !== null && <span className="num">‖g‖ ≤ {opt.clipNorm}</span>}
      </label>
    </div>
  );
}

export function LossHistory({ height = 200 }: { height?: number }) {
  const { t } = useT();
  const history = useLab((s) => s.history);
  const hasVal = history.some((h) => h.val !== undefined);
  const layers = useMemo<PlotLayer[]>(() => {
    const step = Math.max(1, Math.floor(history.length / 600));
    const pts = history.filter((_, i) => i % step === 0 || i === history.length - 1);
    const out: PlotLayer[] = [{ kind: 'polyline', pts: pts.map((h) => [h.iter, h.loss]), color: 'L', width: 2, markers: history.length < 40 }];
    if (hasVal) out.push({ kind: 'polyline', pts: pts.filter((h) => h.val !== undefined).map((h) => [h.iter, h.val!]), color: 'y', width: 2, dash: '6 4' });
    return out;
  }, [history, hasVal]);
  const maxIter = Math.max(5, history[history.length - 1]?.iter ?? 5);
  const maxLoss = Math.max(...history.map((h) => Math.max(h.loss, h.val ?? 0)).filter(Number.isFinite), 1e-6);
  return (
    <Plot2D
      testId="loss-history"
      xDomain={[0, maxIter]}
      yDomain={[0, maxLoss * 1.08]}
      layers={layers}
      height={height}
      xLabel={t('stat.iteration')}
      yLabel={t('stat.loss')}
      legend={hasVal ? [{ label: t('stat.loss'), color: 'L' }, { label: t('stat.valLoss'), color: 'y', dash: true }] : undefined}
    />
  );
}

export function GradCheckPanel() {
  const { t } = useT();
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const loss = useLab((s) => s.loss);
  const lossOpts = useLab((s) => s.lossOpts);
  const [rows, setRows] = useState<GradCheckRow[] | null>(null);
  useEffect(() => setRows(null), [net, data, loss, lossOpts]);
  const worst = rows ? Math.max(0, ...rows.filter((r) => !r.nearKink).map((r) => (Math.abs(r.analytic - r.numeric) < 1e-9 ? 0 : r.relErr))) : 0;
  return (
    <div className="stack" data-testid="gradcheck">
      <div className="row">
        <button className="btn sm" onClick={() => setRows(gradCheck(net, data, loss, lossOpts))} data-testid="btn-gradcheck">
          {t('gc.run')}
        </button>
        {rows && <span className={`small ${worst < 1e-5 ? 'c-pos' : 'c-neg'}`}>{worst < 1e-5 ? t('gc.ok') : `max ${fmt(worst, 2)}`}</span>}
      </div>
      {rows && (
        <div style={{ maxHeight: 220, overflow: 'auto' }}>
          <table className="tbl">
            <thead>
              <tr>
                <th>θ</th>
                <th className="num">{t('gc.analytic')}</th>
                <th className="num">{t('gc.numeric')}</th>
                <th className="num">{t('gc.err')}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.param.id}>
                  <td className="mono">{r.param.id}</td>
                  <td className="num c-g">{fmt(r.analytic, 6)}</td>
                  <td className="num">{fmt(r.numeric, 6)}</td>
                  <td className="num">{r.nearKink ? <span className="c-neg small">{t('gc.kink')}</span> : fmt(Math.abs(r.analytic - r.numeric) < 1e-9 ? 0 : r.relErr, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/** Every parameter with a slider; frozen ones are marked. */
export function ParamList({ showGrads = true }: { showGrads?: boolean }) {
  const { t } = useT();
  const net = useLab((s) => s.net);
  const frozen = useLab((s) => s.frozen);
  const setParam = useLab((s) => s.setParam);
  const select = useLab((s) => s.select);
  const der = useLabDerived();
  const refs = paramRefs(net);
  return (
    <div className="stack param-list" data-testid="param-list">
      {refs.map((p) => (
        <div key={p.id} onFocus={() => select(p.kind === 'W' ? { type: 'weight', layer: p.layer, j: p.j, i: p.i } : { type: 'bias', layer: p.layer, j: p.j })}>
          <Slider
            label={<span className="mono">{p.id}</span>}
            value={getParam(net, p)}
            min={-5}
            max={5}
            step={0.01}
            color={p.kind === 'W' ? 'w' : 'b'}
            frozen={frozen.includes(p.id)}
            frozenLabel={t('net.frozen')}
            onChange={(v) => setParam(p, v)}
            testId={`param-${p.id}`}
          />
          {showGrads && (
            <div className="param-grad small">
              <span className="ltr" dir="ltr">
                <span className="muted">∂L/∂{p.id}</span> <span className="num c-g">{fmt(getGrad(der.full.grads, p), 4)}</span>
              </span>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export function PresetPicker() {
  const { t, tr } = useT();
  const presetId = useLab((s) => s.presetId);
  const loadPreset = useLab((s) => s.loadPreset);
  return (
    <div className="stack">
      <Select<PresetId | 'custom'>
        label={t('nav.preset')}
        value={presetId}
        onChange={(v) => v !== 'custom' && loadPreset(v)}
        options={[
          ...PRESET_IDS.map((id) => ({ value: id as PresetId | 'custom', label: tr(PRESET_TEXT[id].name) })),
          ...(presetId === 'custom' ? [{ value: 'custom' as const, label: 'Custom' }] : []),
        ]}
      />
      {presetId !== 'custom' && <p className="small muted" style={{ margin: 0 }}>{tr(PRESET_TEXT[presetId].outcome)}</p>}
    </div>
  );
}

export function ArchitectureEditor() {
  const { t } = useT();
  const s = useLab();
  const L = s.net.layers.length;
  const sizes = [s.net.inputSize, ...s.net.layers.map((l) => l.b.length)];
  const actOptions = (out: boolean) =>
    (out ? ([...ELEMENTWISE, 'softmax'] as ActivationName[]) : ELEMENTWISE).map((a) => ({ value: a, label: a }));
  return (
    <div className="stack arch" data-testid="arch-editor">
      {s.net.layers.map((layer, l) => {
        const hidden = l < L - 1;
        return (
          <div key={l} className="arch-row">
            <span className="small">
              {hidden ? `${t('net.hidden')} ${l + 1}` : t('net.output')} <span className="num muted">({sizes[l + 1]})</span>
            </span>
            <Select<ActivationName> label={t('net.activation')} value={layer.act} options={actOptions(!hidden)} onChange={(a) => s.setActivation(l, a)} />
            {hidden && (
              <>
                <button className="btn sm icon" aria-label={t('net.addUnit')} title={t('net.addUnit')} disabled={sizes[l + 1] >= LIMITS.maxUnits} onClick={() => s.addUnit(l)} data-testid={`add-unit-${l}`}>
                  +
                </button>
                <button className="btn sm icon" aria-label={t('net.removeUnit')} title={t('net.removeUnit')} disabled={sizes[l + 1] <= 1} onClick={() => s.removeUnit(l, sizes[l + 1] - 1)}>
                  −
                </button>
                <button className="btn sm ghost" onClick={() => s.removeLayer(l)} data-testid={`remove-layer-${l}`}>
                  {t('net.removeLayer')}
                </button>
              </>
            )}
          </div>
        );
      })}
      <div className="row">
        <button className="btn sm" onClick={s.addLayer} disabled={L - 1 >= LIMITS.maxHiddenLayers} data-testid="add-layer">
          + {t('net.addLayer')}
        </button>
        <Select<LossName>
          label={t('net.lossFn')}
          value={s.loss}
          onChange={(l) => s.setLoss(l)}
          options={(['mse', 'mae', 'huber', 'bce', 'cce'] as const).map((v) => ({ value: v, label: v.toUpperCase() }))}
        />
      </div>
      <div className="row">
        <span className="small">{t('net.init')}</span>
        {(['zeros', 'uniform', 'xavier', 'he'] as const).map((n) => (
          <button key={n} className="btn sm" onClick={() => s.reinit(n, (Date.now() % 997) + 1)}>
            {n}
          </button>
        ))}
      </div>
    </div>
  );
}

export function DataEditor() {
  const { t } = useT();
  const data = useLab((s) => s.data);
  const sample = useLab((s) => s.sample);
  const setSample = useLab((s) => s.setSample);
  const setInput = useLab((s) => s.setInput);
  const setTarget = useLab((s) => s.setTarget);
  const setData = useLab((s) => s.setData);
  const net = useLab((s) => s.net);
  const der = useLabDerived();
  return (
    <div className="stack" data-testid="data-editor">
      <div style={{ maxHeight: 240, overflow: 'auto' }}>
        <table className="tbl">
          <thead>
            <tr>
              <th>#</th>
              <th className="c-x">x</th>
              <th className="c-y">y</th>
              <th className="num c-yhat">ŷ</th>
              <th className="num c-L">L</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {data.X.map((x, n) => (
              <tr key={n} className={n === sample ? 'hl' : ''} onClick={() => setSample(n)} style={{ cursor: 'pointer' }}>
                <td>
                  {n + 1}
                  {der.nextIdx.includes(n) && data.X.length > 1 ? ' •' : ''}
                </td>
                <td>
                  <div className="row" style={{ gap: 4, flexWrap: 'nowrap' }}>
                    {x.map((v, i) => (
                      <NumberField key={i} value={v} label={`x${n + 1}.${i + 1}`} onChange={(nv) => setInput(n, i, nv)} />
                    ))}
                  </div>
                </td>
                <td>
                  <div className="row" style={{ gap: 4, flexWrap: 'nowrap' }}>
                    {data.Y[n].map((v, k) => (
                      <NumberField key={k} value={v} label={`y${n + 1}.${k + 1}`} onChange={(nv) => setTarget(n, k, nv)} />
                    ))}
                  </div>
                </td>
                <td className="num c-yhat">{predict(net, x).map((v) => fmt(v, 3)).join(', ')}</td>
                <td className="num c-L">{fmt(der.full.perSample[n], 4)}</td>
                <td>
                  {data.X.length > 1 && (
                    <button className="btn ghost sm" aria-label={t('explore.removeSample')} onClick={(e) => (e.stopPropagation(), setData({ X: data.X.filter((_, i) => i !== n), Y: data.Y.filter((_, i) => i !== n) }))}>
                      ×
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="row">
        <button className="btn sm" disabled={data.X.length >= 32} onClick={() => setData({ X: [...data.X, data.X[data.X.length - 1].slice()], Y: [...data.Y, data.Y[data.Y.length - 1].slice()] })}>
          + {t('explore.addSample')}
        </button>
        {data.X.length > 1 && <span className="small muted">• = {t('batch.next')}</span>}
      </div>
    </div>
  );
}

/** Output of the network as a function of input x1 (other inputs fixed at the selected example). */
export function OutputPlot({ height = 220 }: { height?: number }) {
  const { t } = useT();
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const sample = useLab((s) => s.sample);
  const units = useLab((s) => s.units);
  const xs = data.X.map((x) => x[0]);
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  const pad = Math.max(0.5, (hi - lo) * 0.25);
  const base = data.X[sample];
  const layers: PlotLayer[] = [
    { kind: 'fn', f: (v) => predict(net, [v, ...base.slice(1)])[0], color: 'yhat', width: 2.4 },
    { kind: 'points', color: 'y', pts: data.X.map((x, n) => ({ x: x[0], y: data.Y[n][0], r: n === sample ? 6.5 : 4.5 })) },
    { kind: 'points', color: 'yhat', pts: data.X.map((x) => ({ x: x[0], y: predict(net, x)[0], r: 3.5, hollow: true })) },
  ];
  return (
    <Plot2D
      testId="output-plot"
      xDomain={[lo - pad, hi + pad]}
      yDomain="auto"
      layers={layers}
      height={height}
      xLabel={units ? t(units.x as 'units.outsideTemp') : 'x₁'}
      yLabel={units ? t(units.y as 'units.acPower') : 'ŷ₁'}
      legend={[
        { label: 'ŷ(x)', color: 'yhat' },
        { label: t('stat.target'), color: 'y' },
      ]}
    />
  );
}

