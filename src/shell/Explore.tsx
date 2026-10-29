// Free exploration: the full playground on the shared lab store.

import { useMemo, useState } from 'react';
import { Panel, Workspace } from './Workspace';
import { useT } from '../i18n';
import { useLab, useLabDerived } from '../state/lab';
import { NetworkCanvas } from '../components/NetworkCanvas';
import { Inspector } from '../components/Inspector';
import {
  ArchitectureEditor,
  DataEditor,
  GradCheckPanel,
  LossHistory,
  OptimizerControls,
  OutputPlot,
  ParamList,
  PresetPicker,
  RunControls,
  TrainStats,
} from '../components/LabPanels';
import { Segmented, Select } from '../components/controls';
import { Plot2D } from '../components/Plot2D';
import { datasetLoss, fmt, getGrad, getParam, paramRefs, setParam } from '../engine';

export function ExploreMode() {
  const { t } = useT();
  const [tab, setTab] = useState<'output' | 'history' | 'slice' | 'check'>('history');
  return (
    <Workspace
      left={
        <Panel id="left" title="explore.title">
          <section className="stack">
            <h3>{t('nav.preset')}</h3>
            <PresetPicker />
            <h3>{t('explore.network')}</h3>
            <ArchitectureEditor />
            <h3>{t('explore.data')}</h3>
            <DataEditor />
            <h3>{t('explore.training')}</h3>
            <OptimizerControls />
          </section>
        </Panel>
      }
      center={
        <Panel id="center" title="panel.visual" maximizable>
          <div className="center-stack">
            <RunControls />
            <div className="canvas-wrap">
              <NetworkCanvas />
            </div>
            <TrainStats />
          </div>
        </Panel>
      }
      right={
        <Panel id="right" title="panel.math" maximizable>
          <div className="stack">
            <Inspector />
            <details open>
              <summary className="small">{t('stat.weight')} / {t('stat.bias')}</summary>
              <ParamList />
            </details>
          </div>
        </Panel>
      }
      bottom={
        <Panel
          id="bottom"
          title="panel.graphs"
          actions={
            <Segmented
              label={t('panel.graphs')}
              value={tab}
              onChange={setTab}
              options={[
                { value: 'history', label: t('explore.history') },
                { value: 'output', label: t('explore.output') },
                { value: 'slice', label: t('explore.landscape') },
                { value: 'check', label: t('gc.title') },
              ]}
            />
          }
        >
          {tab === 'history' && <LossHistory />}
          {tab === 'output' && <OutputPlot />}
          {tab === 'slice' && <LossSlice />}
          {tab === 'check' && <GradCheckPanel />}
        </Panel>
      }
    />
  );
}

/** Loss along one parameter with every other parameter held at its current value. */
export function LossSlice({ height = 220 }: { height?: number }) {
  const { t } = useT();
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const loss = useLab((s) => s.loss);
  const lossOpts = useLab((s) => s.lossOpts);
  const selection = useLab((s) => s.selection);
  const refs = paramRefs(net);
  const der = useLabDerived();
  const fromSel = selection?.type === 'weight' ? `W${selection.layer + 1}[${selection.j},${selection.i}]` : selection?.type === 'bias' ? `b${selection.layer + 1}[${selection.j}]` : null;
  const [pick, setPick] = useState(refs[0].id);
  const id = fromSel && refs.some((r) => r.id === fromSel) ? fromSel : refs.some((r) => r.id === pick) ? pick : refs[0].id;
  const ref = refs.find((r) => r.id === id)!;
  const v = getParam(net, ref);
  const g = getGrad(der.full.grads, ref);
  const f = useMemo(() => (p: number) => datasetLoss(setParam(net, ref, p), data, loss, lossOpts), [net, ref, data, loss, lossOpts]);
  const L0 = f(v);
  return (
    <div className="stack">
      <div className="row">
        <Select label="θ" value={id} onChange={setPick} options={refs.map((r) => ({ value: r.id, label: r.id }))} />
        <span className="small muted">{t('explore.sliceNote')}</span>
      </div>
      <Plot2D
        testId="loss-slice"
        xDomain={[v - 3, v + 3]}
        yDomain="auto"
        height={height}
        xLabel={id}
        yLabel="L"
        layers={[
          { kind: 'fn', f, color: 'L', width: 2.4 },
          { kind: 'line', p: [v, L0], slope: g, color: 'g', dash: '6 4' },
          { kind: 'points', color: 'L', pts: [{ x: v, y: L0, label: `L ${fmt(L0, 3)}, ∂L/∂θ ${fmt(g, 3)}` }] },
        ]}
      />
    </div>
  );
}
