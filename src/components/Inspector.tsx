// Details for the selected neuron, connection, bias or loss: live equations, the chain-rule
// paths behind its gradient, and a nudge test that re-runs the forward pass.

import { useMemo } from 'react';
import { useLab, useLabDerived } from '../state/lab';
import { useT } from '../i18n';
import { Tex } from './Tex';
import { Slider } from './controls';
import { actDeriv, fmt, lossSample, pathsFrom, predict, setParam, sumContrib, tex, texP, type ActivationName, type LossName } from '../engine';

const F: Record<ActivationName, string> = {
  linear: '',
  relu: '\\relu',
  sigmoid: '\\sigma',
  tanh: '\\tanh',
  softmax: '\\operatorname{softmax}',
};

export const ACT_DERIV_TEX: Record<ActivationName, string> = {
  linear: "f'(z) = 1",
  relu: "\\relu'(z) = \\begin{cases}1 & z>0\\\\0 & z\\le 0\\ (\\text{convention})\\end{cases}",
  sigmoid: "\\sigma'(z) = \\sigma(z)\\,(1-\\sigma(z))",
  tanh: "\\tanh'(z) = 1-\\tanh^2(z)",
  softmax: '\\partial a_i/\\partial z_k = a_i(\\delta_{ik}-a_k)',
};

export const LOSS_TEX: Record<LossName, string> = {
  mse: '\\cL{L} = \\frac{1}{K}\\sum_k (\\cyhat{\\hat y_k}-\\cy{y_k})^2',
  mae: '\\cL{L} = \\frac{1}{K}\\sum_k |\\cyhat{\\hat y_k}-\\cy{y_k}|',
  huber: '\\cL{L} = \\frac{1}{K}\\sum_k \\begin{cases}\\tfrac12 r_k^2 & |r_k|\\le\\delta\\\\ \\delta(|r_k|-\\tfrac12\\delta) & \\text{else}\\end{cases}',
  bce: '\\cL{L} = -\\frac{1}{K}\\sum_k \\big[\\cy{y_k}\\log\\cyhat{\\hat y_k} + (1-\\cy{y_k})\\log(1-\\cyhat{\\hat y_k})\\big]',
  cce: '\\cL{L} = -\\sum_k \\cy{y_k}\\log\\cyhat{\\hat y_k}',
};

const H = 1e-4;

export function Inspector() {
  const { t } = useT();
  const sel = useLab((s) => s.selection);
  const net = useLab((s) => s.net);
  const data = useLab((s) => s.data);
  const loss = useLab((s) => s.loss);
  const lossOpts = useLab((s) => s.lossOpts);
  const frozen = useLab((s) => s.frozen);
  const setParamStore = useLab((s) => s.setParam);
  const der = useLabDerived();
  const { one, sample } = der;
  const x = data.X[sample];
  const y = data.Y[sample];
  const L = net.layers.length;
  const multi = data.X.length > 1;

  const content = useMemo(() => {
    if (!sel) return null;
    const lossAt = (n2: typeof net) => lossSample(loss, predict(n2, x), y, lossOpts);
    if (sel.type === 'loss') {
      const out = one.trace.a[L];
      return (
        <>
          <h4>{t('net.loss')}</h4>
          <Tex display>{LOSS_TEX[loss]}</Tex>
          <Tex display>{`\\cL{L} = ${tex(one.loss, 4)},\\quad \\cyhat{\\hat y} = (${out.map((v) => tex(v)).join(', ')}),\\ \\cy{y} = (${y.map((v) => tex(v)).join(', ')})`}</Tex>
          <Tex display>{`\\cg{\\frac{\\partial L}{\\partial \\hat y}} = (${one.dOut.map((v) => tex(v)).join(', ')})`}</Tex>
        </>
      );
    }
    if (sel.type === 'unit') {
      const l = sel.layer;
      const k = sel.unit;
      if (l === 0)
        return (
          <>
            <h4>
              {t('net.input')} x<sub>{k + 1}</sub>
            </h4>
            <Tex display>{`\\cx{x_{${k + 1}}} = ${tex(x[k])}`}</Tex>
            <Tex display>{`\\cg{\\frac{\\partial L}{\\partial x_{${k + 1}}}} = ${tex(one.back.da[0][k], 4)}`}</Tex>
          </>
        );
      const layer = net.layers[l - 1];
      const prev = one.trace.a[l - 1];
      const z = one.trace.z[l - 1][k];
      const a = one.trace.a[l][k];
      const prevSym = l - 1 === 0 ? 'x' : 'a^{(' + (l - 1) + ')}';
      const sumSym = layer.W[k].map((_, i) => `\\cw{w_{${k + 1}${i + 1}}}\\,\\ca{${prevSym}_{${i + 1}}}`).join(' + ');
      const sumNum = layer.W[k].map((w, i) => `${texP(w)}\\cdot${texP(prev[i])}`).join(' + ');
      const d = layer.act === 'softmax' ? NaN : actDeriv(layer.act, z);
      const paths = pathsFrom(net, one.trace, one.back.dz, l - 1, k);
      return (
        <>
          <h4>
            {t('insp.unit')} {l}.{k + 1} <span className="muted small">({layer.act})</span>
          </h4>
          <Tex display>{`\\cz{z} = ${sumSym} + \\cb{b_{${k + 1}}}`}</Tex>
          <Tex display>{`\\cz{z} = ${sumNum} + ${texP(layer.b[k])} = ${tex(z, 4)}`}</Tex>
          <Tex display>{`\\ca{a} = ${F[layer.act]}${layer.act === 'linear' ? '' : '('}\\cz{${tex(z, 3)}}${layer.act === 'linear' ? '' : ')'} = ${tex(a, 4)}`}</Tex>
          {layer.act !== 'softmax' && <Tex display>{`${ACT_DERIV_TEX[layer.act]} \\Rightarrow ${tex(d, 4)}`}</Tex>}
          {layer.act === 'relu' && z === 0 && <div className="callout warn small">ReLU'(0): {t('gc.kink')}. Engine uses 0.</div>}
          <Tex display>{`\\cg{\\frac{\\partial L}{\\partial z}} = ${tex(one.back.dz[l - 1][k], 4)}`}</Tex>
          {l < L && <PathList paths={paths} total={one.back.dz[l - 1][k]} />}
        </>
      );
    }
    if (sel.type === 'weight' || sel.type === 'bias') {
      const isW = sel.type === 'weight';
      const l = sel.layer;
      const j = sel.j;
      const i = isW ? sel.i : -1;
      const ref = { kind: isW ? ('W' as const) : ('b' as const), layer: l, j, i };
      const id = isW ? `W${l + 1}[${j},${i}]` : `b${l + 1}[${j}]`;
      const v = isW ? net.layers[l].W[j][i] : net.layers[l].b[j];
      const g = isW ? one.back.grads.W[l][j][i] : one.back.grads.b[l][j];
      const ain = isW ? one.trace.a[l][i] : 1;
      const dz = one.back.dz[l][j];
      const nudged = (lossAt(setParam(net, { ...ref, id }, v + H)) - lossAt(setParam(net, { ...ref, id }, v - H))) / (2 * H);
      const paths = pathsFrom(net, one.trace, one.back.dz, l, j, isW ? ain : undefined);
      const inSym = l === 0 ? `\\cx{x_{${i + 1}}}` : `\\ca{a^{(${l})}_{${i + 1}}}`;
      const isFrozen = frozen.includes(id);
      return (
        <>
          <h4>
            {isW ? t('insp.weight') : t('insp.bias')} <span className="mono">{id}</span>
          </h4>
          <Slider
            label={isW ? `w = ${id}` : `b = ${id}`}
            value={v}
            min={Math.min(-5, v - 1)}
            max={Math.max(5, v + 1)}
            step={0.01}
            color={isW ? 'w' : 'b'}
            frozen={isFrozen}
            frozenLabel={t('net.frozen')}
            onChange={(nv) => setParamStore(ref, nv)}
            testId="inspector-slider"
          />
          {isW ? (
            <Tex display>{`\\cg{\\frac{\\partial L}{\\partial w}} = \\cg{\\frac{\\partial L}{\\partial z_{${j + 1}}}}\\cdot ${inSym} = ${texP(dz, 4)}\\cdot${texP(ain)} = ${tex(g, 4)}`}</Tex>
          ) : (
            <Tex display>{`\\cg{\\frac{\\partial L}{\\partial b}} = \\cg{\\frac{\\partial L}{\\partial z_{${j + 1}}}}\\cdot 1 = ${tex(g, 4)}`}</Tex>
          )}
          <PathList paths={paths} total={g} />
          <div className="nudge">
            <div className="small muted">{t('insp.nudge')} (h = 10⁻⁴)</div>
            <Tex display>{`\\frac{L(${isW ? 'w' : 'b'}+h)-L(${isW ? 'w' : 'b'}-h)}{2h} = ${tex(nudged, 4)} \\quad\\text{vs backprop } ${tex(g, 4)}`}</Tex>
          </div>
        </>
      );
    }
    return null;
  }, [sel, net, one, x, y, loss, lossOpts, L, t, frozen, setParamStore]);

  return (
    <div className="inspector" data-testid="inspector">
      {multi && sel && (
        <div className="muted small" style={{ marginBottom: 6 }}>
          {t('net.sample')} #{sample + 1}
        </div>
      )}
      {content ?? <p className="muted">{t('net.selectHint')}</p>}
    </div>
  );
}

function PathList({ paths, total }: { paths: ReturnType<typeof pathsFrom>; total: number }) {
  const { t } = useT();
  if (!paths.length) return null;
  const sum = sumContrib(paths);
  return (
    <div className="paths" data-testid="path-list">
      <div className="small muted">
        {t('insp.paths')} ({paths.length}) — {t('insp.pathsNote')}
      </div>
      {paths.slice(0, 12).map((p, k) => (
        <div key={k} className="path-row">
          <Tex>{`${p.factors.map((f) => texP(f)).join('\\times ')} = ${tex(p.contribution, 4)}`}</Tex>
        </div>
      ))}
      {paths.length > 12 && <div className="small muted">+{paths.length - 12}</div>}
      <div className="path-sum">
        <Tex>{`\\textstyle\\sum = ${tex(sum, 4)}`}</Tex>
        <span className="small muted">
          {' '}
          {t('insp.backprop')}: <span className="num c-g">{fmt(total, 4)}</span>
        </span>
      </div>
    </div>
  );
}
