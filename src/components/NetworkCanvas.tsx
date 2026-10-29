// The network diagram (React Flow). Reads the shared lab store only: every value shown is the
// engine's value for the selected training example. Edges are colored by weight sign and sized
// by |w|; the forward/backward stepper controls what is revealed and which edges pulse.

import { memo, useEffect, useMemo, useRef } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  Handle,
  Position,
  BaseEdge,
  EdgeLabelRenderer,
  getStraightPath,
  type Edge,
  type EdgeProps,
  type Node,
  type NodeProps,
  useReactFlow,
  useNodesInitialized,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useLab, useLabDerived, visibility, type Selection } from '../state/lab';
import { pathsFrom, fmt, type ActivationName } from '../engine';
import { useT } from '../i18n';
import { useUi } from '../state/ui';

type UnitData = {
  layer: number;
  unit: number;
  kind: 'input' | 'hidden' | 'output';
  act?: ActivationName;
  z?: number;
  a?: number;
  showA: boolean;
  grad?: number;
  showGrad: boolean;
  selected: boolean;
  onPath: boolean;
  dead: boolean;
  active: boolean;
  flash: boolean;
  label: string;
};

type LossData = { loss: number; show: boolean; target: number[]; dOut: number[]; showGrad: boolean; selected: boolean; label: string };

type WeightData = {
  w: number;
  g?: number;
  showGrad: boolean;
  selected: boolean;
  onPath: boolean;
  pulse: 'fwd' | 'bwd' | null;
  maxAbs: number;
  frozen: boolean;
  flash: boolean;
  label: string;
};

const ACT_SYMBOL: Record<ActivationName, string> = { linear: 'id', relu: 'ReLU', sigmoid: 'σ', tanh: 'tanh', softmax: 'softmax' };

/** Briefly highlights its parent when `value` changes after mount, to show cause and effect. */
function Flash({ value, enabled }: { value: string; enabled: boolean }) {
  const initial = useRef(value);
  if (!enabled || initial.current === value) return null;
  return <span key={value} className="nn-flash" aria-hidden="true" />;
}

const UnitNode = memo(function UnitNode({ data }: NodeProps<Node<UnitData>>) {
  const d = data;
  return (
    <div
      className={`nn-unit ${d.kind} ${d.selected ? 'sel' : ''} ${d.onPath ? 'path' : ''} ${d.dead ? 'dead' : ''} ${d.active ? 'active' : ''}`}
      data-testid={`unit-${d.layer}-${d.unit}`}
      aria-label={d.label}
    >
      {d.kind !== 'input' && <Handle type="target" position={Position.Left} />}
      {d.kind !== 'output' && <Handle type="source" position={Position.Right} />}
      {d.kind === 'output' && <Handle type="source" position={Position.Right} id="out" />}
      <Flash value={`${fmt(d.a ?? 0, 4)}`} enabled={d.flash && d.showA} />
      {d.act && <div className="nn-act">{ACT_SYMBOL[d.act]}</div>}
      <div className="nn-vals">
        {d.kind !== 'input' && <span className="c-z">{d.showA && d.z !== undefined ? `z ${fmt(d.z, 2)}` : 'z ?'}</span>}
        <span className={d.kind === 'input' ? 'c-x' : d.kind === 'output' ? 'c-yhat' : 'c-a'}>
          {d.kind === 'input' ? 'x' : d.kind === 'output' ? 'ŷ' : 'a'} {d.showA && d.a !== undefined ? fmt(d.a, 2) : '?'}
        </span>
      </div>
      {d.showGrad && d.grad !== undefined && <div className="nn-grad c-g">∂L/∂z {fmt(d.grad, 2)}</div>}
    </div>
  );
});

const LossNode = memo(function LossNode({ data }: NodeProps<Node<LossData>>) {
  const d = data;
  return (
    <div className={`nn-loss ${d.selected ? 'sel' : ''}`} data-testid="loss-node" aria-label={d.label}>
      <Handle type="target" position={Position.Left} />
      <div className="c-L nn-loss-v">L {d.show ? fmt(d.loss, 3) : '?'}</div>
      <div className="c-y small">y {d.target.map((v) => fmt(v, 2)).join(', ')}</div>
      {d.showGrad && <div className="c-g small">∂L/∂ŷ {d.dOut.map((v) => fmt(v, 2)).join(', ')}</div>}
    </div>
  );
});

const WeightEdge = memo(function WeightEdge(props: EdgeProps<Edge<WeightData>>) {
  const { sourceX, sourceY, targetX, targetY, data, id } = props;
  const d = data!;
  const [path, lx, ly] = getStraightPath({ sourceX, sourceY, targetX, targetY });
  const width = 1.2 + 5 * Math.min(1, Math.abs(d.w) / Math.max(1e-9, d.maxAbs));
  const color = d.w >= 0 ? 'var(--c-wpos)' : 'var(--c-wneg)';
  return (
    <>
      {(d.selected || d.onPath) && <BaseEdge id={id + '-halo'} path={path} style={{ stroke: 'var(--c-g)', strokeWidth: width + 7, opacity: 0.22 }} />}
      <BaseEdge id={id} path={path} style={{ stroke: color, strokeWidth: width, strokeDasharray: d.frozen ? '5 4' : undefined, opacity: d.w === 0 ? 0.35 : 0.9 }} interactionWidth={18} />
      {d.pulse && (
        <circle r={5} fill={d.pulse === 'fwd' ? 'var(--c-x)' : 'var(--c-g)'} className="nn-pulse">
          <animateMotion dur="0.9s" repeatCount="indefinite" path={path} keyPoints={d.pulse === 'fwd' ? '0;1' : '1;0'} keyTimes="0;1" calcMode="linear" />
        </circle>
      )}
      <EdgeLabelRenderer>
        <div
          className={`nn-wlabel ${d.selected ? 'sel' : ''}`}
          style={{ transform: `translate(-50%, -50%) translate(${lx}px, ${ly}px)` }}
          data-testid={`edge-label-${id}`}
          aria-label={d.label}
        >
          <Flash value={fmt(d.w, 4)} enabled={d.flash} />
          <span className="c-w">{fmt(d.w, 2)}</span>
          {d.showGrad && d.g !== undefined && <span className="c-g"> ∂ {fmt(d.g, 2)}</span>}
        </div>
      </EdgeLabelRenderer>
    </>
  );
});

const nodeTypes = { unit: UnitNode, loss: LossNode };

/** Refit the view when nodes are first measured and whenever the panel is resized. */
function AutoFit({ target }: { target: React.RefObject<HTMLDivElement | null> }) {
  const { fitView } = useReactFlow();
  const ready = useNodesInitialized();
  useEffect(() => {
    if (ready) fitView({ padding: 0.18, maxZoom: 1.4 });
  }, [ready, fitView]);
  useEffect(() => {
    const el = target.current;
    if (!el) return;
    let raf = 0;
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => fitView({ padding: 0.18, maxZoom: 1.4 }));
    });
    ro.observe(el);
    return () => (ro.disconnect(), cancelAnimationFrame(raf));
  }, [target, fitView]);
  return null;
}
const edgeTypes = { weight: WeightEdge };

const sameSel = (a: Selection, b: Selection) => JSON.stringify(a) === JSON.stringify(b);

export function NetworkCanvas({ height = '100%', showLoss = true, interactive = true }: { height?: number | string; showLoss?: boolean; interactive?: boolean }) {
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const selection = useLab((s) => s.selection);
  const stepper = useLab((s) => s.stepper);
  const frozen = useLab((s) => s.frozen);
  const select = useLab((s) => s.select);
  const reduced = useUi((s) => s.reducedMotion);
  const running = useLab((s) => s.running);
  const wrap = useRef<HTMLDivElement>(null);
  const { t } = useT();
  const der = useLabDerived();
  const { one } = der;
  const L = net.layers.length;
  const vis = visibility(stepper, L);
  const sizesArr = der.sizes;

  // Units/weights on the gradient path of the selected item (for highlighting).
  const onPath = useMemo(() => {
    const units = new Set<string>();
    const edges = new Set<string>();
    let start: { l: number; j: number } | null = null;
    if (selection?.type === 'weight') {
      start = { l: selection.layer, j: selection.j };
      edges.add(`w-${selection.layer}-${selection.j}-${selection.i}`);
    } else if (selection?.type === 'bias') start = { l: selection.layer, j: selection.j };
    else if (selection?.type === 'unit' && selection.layer > 0) start = { l: selection.layer - 1, j: selection.unit };
    if (start) {
      for (const p of pathsFrom(net, one.trace, one.back.dz, start.l, start.j)) {
        p.steps.forEach((s) => units.add(`${s.layer + 1}-${s.unit}`));
        p.steps.forEach((s) => s.weight && edges.add(`w-${s.weight.layer}-${s.weight.j}-${s.weight.i}`));
      }
    }
    return { units, edges };
  }, [selection, net, one]);

  const { nodes, edges } = useMemo(() => {
    const colW = 230;
    const rowH = 122;
    const maxN = Math.max(...sizesArr);
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    const y = (n: number, k: number) => (maxN - n) * (rowH / 2) + k * rowH;
    const maxAbs = Math.max(1e-9, ...net.layers.flatMap((l) => l.W.flat().map(Math.abs)));
    sizesArr.forEach((n, l) => {
      for (let k = 0; k < n; k++) {
        const kind = l === 0 ? 'input' : l === L ? 'output' : 'hidden';
        const z = l > 0 ? one.trace.z[l - 1][k] : undefined;
        const nd: UnitData = {
          layer: l,
          unit: k,
          kind,
          act: l > 0 ? net.layers[l - 1].act : undefined,
          z,
          a: one.trace.a[l][k],
          showA: vis.act(l),
          grad: l > 0 ? one.back.dz[l - 1][k] : undefined,
          showGrad: l > 0 && vis.grad(l),
          selected: selection?.type === 'unit' && selection.layer === l && selection.unit === k,
          onPath: onPath.units.has(`${l}-${k}`),
          dead: l > 0 && net.layers[l - 1].act === 'relu' && (z ?? 1) <= 0 && vis.act(l),
          flash: !running && !reduced,
          active: stepper?.phase === 'forward' ? stepper.k - 1 === l : stepper?.phase === 'backward' ? l > 0 && L + 1 - l === stepper.k - 1 : false,
          label: `${t(kind === 'input' ? 'net.input' : kind === 'output' ? 'net.output' : 'net.hidden')} ${l}.${k + 1}`,
        };
        nodes.push({ id: `u-${l}-${k}`, type: 'unit', position: { x: l * colW, y: y(n, k) }, data: nd, draggable: false });
      }
    });
    if (showLoss) {
      const n = sizesArr[L];
      nodes.push({
        id: 'loss',
        type: 'loss',
        position: { x: L * colW + 170, y: y(n, 0) + ((n - 1) * rowH) / 2 },
        draggable: false,
        data: {
          loss: one.loss,
          show: vis.loss,
          target: data.Y[der.sample],
          dOut: one.dOut,
          showGrad: vis.dOut,
          selected: selection?.type === 'loss',
          label: t('net.loss'),
        } satisfies LossData,
      });
      for (let k = 0; k < n; k++) edges.push({ id: `o-${k}`, source: `u-${L}-${k}`, sourceHandle: 'out', target: 'loss', style: { stroke: 'var(--line-strong)', strokeDasharray: '3 4' }, selectable: false });
    }
    net.layers.forEach((layer, l) =>
      layer.W.forEach((row, j) =>
        row.forEach((w, i) => {
          const id = `w-${l}-${j}-${i}`;
          const pulse: WeightData['pulse'] =
            reduced || !stepper ? null : stepper.phase === 'forward' && stepper.k - 1 === l + 1 ? 'fwd' : stepper.phase === 'backward' && L + 1 - (l + 1) === stepper.k - 1 ? 'bwd' : null;
          edges.push({
            id,
            source: `u-${l}-${i}`,
            target: `u-${l + 1}-${j}`,
            type: 'weight',
            data: {
              w,
              g: one.back.grads.W[l][j][i],
              showGrad: vis.grad(l + 1),
              selected: selection?.type === 'weight' && selection.layer === l && selection.j === j && selection.i === i,
              onPath: onPath.edges.has(id),
              pulse,
              maxAbs,
              frozen: frozen.includes(`W${l + 1}[${j},${i}]`),
              flash: !running && !reduced,
              label: `W${l + 1}[${j},${i}] = ${fmt(w, 3)}`,
            } satisfies WeightData,
          });
        }),
      ),
    );
    return { nodes, edges };
  }, [net, one, data, der.sample, sizesArr, L, vis.loss, vis.dOut, stepper, selection, onPath, frozen, showLoss, reduced, running, t]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="nn-canvas" style={{ height }} dir="ltr" data-testid="network-canvas" ref={wrap}>
      <ReactFlow
        key={sizesArr.join('-') + (showLoss ? 'L' : '')}
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        minZoom={0.3}
        maxZoom={2.5}
        nodesConnectable={false}
        nodesDraggable={false}
        elementsSelectable={interactive}
        proOptions={{ hideAttribution: true }}
        onNodeClick={(_, n) => {
          if (!interactive) return;
          const s: Selection = n.id === 'loss' ? { type: 'loss' } : { type: 'unit', layer: (n.data as UnitData).layer, unit: (n.data as UnitData).unit };
          select(sameSel(s, selection) ? null : s);
        }}
        onEdgeClick={(_, e) => {
          if (!interactive || !e.id.startsWith('w-')) return;
          const [, l, j, i] = e.id.split('-').map(Number);
          const s: Selection = { type: 'weight', layer: l, j, i };
          select(sameSel(s, selection) ? null : s);
        }}
        onPaneClick={() => interactive && select(null)}
      >
        <AutoFit target={wrap} />
        <Background gap={24} size={1} color="var(--line)" />
        <Controls showInteractive={false} position="bottom-right" />
      </ReactFlow>
    </div>
  );
}
