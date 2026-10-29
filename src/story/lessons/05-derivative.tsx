// Lesson 5 · Slope = derivative. How fast the loss changes when you nudge a knob: the slope of
// lesson 2, now at one point of a curve. Script: docs/storyboard.md, lesson 5.
//
// One plot carries the whole lesson. Its camera (the visible w and L ranges) is animated, so the
// zoom at w = 3 is continuous: the curve really straightens as the view shrinks around the point.

import { useEffect, useState, type ReactNode } from 'react';
import { bi, useT } from '../../i18n';
import { fmt, niceTicks, tex } from '../../engine';
import { useLessonState } from '../../state/lessonState';
import { Tex } from '../../components/Tex';
import type { StoryLesson } from '../types';
import { Appear, C, ValueCard } from '../kit';
import { useTween, useTicker } from '../useTween';
import {
  LINE_TRIANGLES,
  LINE_VIEW,
  LINE_W,
  SIGN_DEMO,
  TANGENT_AT,
  W0,
  W_BEST,
  W_TURN,
  Z_NUDGE,
  Z_STEPS,
  line,
  loss,
  mixView,
  nudge,
  slope,
  stepDigits,
  stepTriangle,
  zoomView,
  type View,
} from './math-derivative';

const ID = 'derivative';
const NUDGED = nudge(W0);
/** The loss meter's full scale: L(0) = L(10) = 100, so the needle sweeps the whole dial with the U. */
const METER_MAX = loss(0);
const W_MIN = 0;
const W_MAX = 10;
/** Step 5's triangle on the ×100 view, before we measure it. */
const PEEK_RUN = 0.02;

// ---- Knob and meter without their own easing ------------------------------------------------
// Same markup as kit's Knob and Meter. The scene eases the value itself, so the dot on the curve,
// the knob and the needle always move together (also while sweeping and dragging).

function LiveKnob({ value, digits, dim }: { value: number; digits: number; dim: boolean }) {
  const size = 118;
  const t = Math.max(0, Math.min(1, (value - W_MIN) / (W_MAX - W_MIN)));
  const ang = -135 + 270 * t;
  const r = size / 2 - 8;
  const cx = size / 2;
  const p = (a: number) => [cx + r * Math.sin((a * Math.PI) / 180), cx - r * Math.cos((a * Math.PI) / 180)];
  const arc = (a0: number, a1: number) => {
    const [x0, y0] = p(a0);
    const [x1, y1] = p(a1);
    return `M${x0},${y0} A${r},${r} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${x1},${y1}`;
  };
  return (
    <div className="knob" dir="ltr" style={{ opacity: dim ? 0.35 : 1, transition: 'opacity 0.5s' }}>
      <svg width={size} height={size} role="img" aria-label={`knob ${fmt(value, digits)}`}>
        <path d={arc(-135, 135)} stroke="var(--line-strong)" strokeWidth={6} fill="none" strokeLinecap="round" />
        <path d={arc(-135, ang)} stroke={C.weight} strokeWidth={6} fill="none" strokeLinecap="round" />
        <circle cx={cx} cy={cx} r={r - 12} fill="var(--surface)" stroke="var(--line-strong)" />
        <line x1={cx} y1={cx} x2={cx + (r - 16) * Math.sin((ang * Math.PI) / 180)} y2={cx - (r - 16) * Math.cos((ang * Math.PI) / 180)} stroke={C.weight} strokeWidth={4} strokeLinecap="round" />
      </svg>
      <div className="knob-label" style={{ color: C.weight }}>
        w
      </div>
      <div className="knob-value num" style={{ fontSize: '1.5em' }}>
        {fmt(value, digits)}
      </div>
    </div>
  );
}

function LiveMeter({ value, digits, dim }: { value: number; digits: number; dim: boolean }) {
  const t = Math.max(0, Math.min(1, value / METER_MAX));
  const ang = -90 + 180 * t;
  const W = 150;
  const r = 60;
  const cx = W / 2;
  const cy = 72;
  const sin = Math.sin((ang * Math.PI) / 180);
  const cos = Math.cos((ang * Math.PI) / 180);
  return (
    <div className="meter" dir="ltr" style={{ opacity: dim ? 0.35 : 1, transition: 'opacity 0.5s' }}>
      <svg width={W} height={84} role="img" aria-label={`meter ${fmt(value, digits)}`}>
        <path d={`M${cx - r},${cy} A${r},${r} 0 0 1 ${cx + r},${cy}`} stroke="var(--line-strong)" strokeWidth={10} fill="none" />
        <path d={`M${cx - r},${cy} A${r},${r} 0 0 1 ${cx + r * sin},${cy - r * cos}`} stroke={C.loss} strokeWidth={10} fill="none" />
        <line x1={cx} y1={cy} x2={cx + (r - 14) * sin} y2={cy - (r - 14) * cos} stroke="var(--ink)" strokeWidth={3} strokeLinecap="round" />
        <circle cx={cx} cy={cy} r={5} fill="var(--ink)" />
      </svg>
      <div className="meter-label" style={{ color: C.loss }}>
        L
      </div>
      <div className="meter-value num" style={{ color: C.loss, fontSize: '1.5em' }}>
        {fmt(value, digits)}
      </div>
    </div>
  );
}

// ---- The plot --------------------------------------------------------------------------------

const PW = 880;
const PH = 480;
const PAD = { l: 92, r: 22, t: 18, b: 52 };

interface Frame {
  v: View;
  sx: (x: number) => number;
  sy: (y: number) => number;
  /** Pixels per unit, to draw things with a fixed on-screen size. */
  kx: number;
  ky: number;
}

const frameOf = (v: View): Frame => {
  const kx = (PW - PAD.l - PAD.r) / (v.x1 - v.x0);
  const ky = (PH - PAD.t - PAD.b) / (v.y1 - v.y0);
  return { v, kx, ky, sx: (x) => PAD.l + (x - v.x0) * kx, sy: (y) => PH - PAD.b - (y - v.y0) * ky };
};

/** A function's path over [a, b] ∩ view, sampled finely enough at any zoom. */
function fnPath(F: Frame, f: (x: number) => number, a: number, b: number, n = 360) {
  const lo = Math.max(a, F.v.x0);
  const hi = Math.min(b, F.v.x1);
  if (!(hi > lo)) return '';
  const span = F.v.y1 - F.v.y0;
  const clip = (y: number) => Math.max(F.v.y0 - 3 * span, Math.min(F.v.y1 + 3 * span, y));
  let d = '';
  for (let i = 0; i <= n; i++) {
    const x = lo + ((hi - lo) * i) / n;
    d += `${i ? 'L' : 'M'}${F.sx(x).toFixed(2)},${F.sy(clip(f(x))).toFixed(2)}`;
  }
  return d;
}

const label = (v: number, d: number) => v.toFixed(d).replace('-', '−');

function Axes({ F, xName, yName, xColor, yColor }: { F: Frame; xName: string; yName: string; xColor: string; yColor: string }) {
  const { v } = F;
  const xt = niceTicks(v.x0, v.x1, 7);
  const yt = niceTicks(v.y0, v.y1, 6);
  const dx = xt.length > 1 ? stepDigits(+(xt[1] - xt[0]).toPrecision(6)) : 0;
  const dy = yt.length > 1 ? stepDigits(+(yt[1] - yt[0]).toPrecision(6)) : 0;
  return (
    <g>
      {xt.map((t) => (
        <g key={'x' + t}>
          <line x1={F.sx(t)} x2={F.sx(t)} y1={PAD.t} y2={PH - PAD.b} stroke="var(--line)" strokeWidth={1} />
          <text x={F.sx(t)} y={PH - PAD.b + 22} textAnchor="middle" fontSize={15} fill="var(--muted)">
            {label(t, dx)}
          </text>
        </g>
      ))}
      {yt.map((t) => (
        <g key={'y' + t}>
          <line x1={PAD.l} x2={PW - PAD.r} y1={F.sy(t)} y2={F.sy(t)} stroke="var(--line)" strokeWidth={1} />
          <text x={PAD.l - 8} y={F.sy(t) + 5} textAnchor="end" fontSize={15} fill="var(--muted)">
            {label(t, dy)}
          </text>
        </g>
      ))}
      {v.y0 < 0 && v.y1 > 0 && <line x1={PAD.l} x2={PW - PAD.r} y1={F.sy(0)} y2={F.sy(0)} stroke="var(--line-strong)" strokeWidth={1.5} />}
      <rect x={PAD.l} y={PAD.t} width={PW - PAD.l - PAD.r} height={PH - PAD.t - PAD.b} fill="none" stroke="var(--line-strong)" />
      <text x={PW - PAD.r} y={PH - 6} textAnchor="end" fontSize={22} fontWeight={700} fill={xColor}>
        {xName} →
      </text>
      <text x={8} y={PAD.t + 18} fontSize={22} fontWeight={700} fill={yColor}>
        {yName}
      </text>
      <text x={8} y={PAD.t + 40} fontSize={18} fill={yColor}>
        ↑
      </text>
    </g>
  );
}

/** A tangent piece of fixed on-screen half-length through (x, f(x)) with slope m. */
function tangentEnds(F: Frame, x: number, y: number, m: number, half: number) {
  const ux = F.kx;
  const uy = -m * F.ky;
  const n = Math.hypot(ux, uy);
  const cx = F.sx(x);
  const cy = F.sy(y);
  return { x1: cx - (half * ux) / n, y1: cy - (half * uy) / n, x2: cx + (half * ux) / n, y2: cy + (half * uy) / n };
}

function Dot({ F, x, y, color = C.weight, r = 9, halo = false }: { F: Frame; x: number; y: number; color?: string; r?: number; halo?: boolean }) {
  return (
    <g>
      {halo && <circle cx={F.sx(x)} cy={F.sy(y)} r={r + 9} fill={color} opacity={0.15} />}
      <circle cx={F.sx(x)} cy={F.sy(y)} r={r} fill={color} stroke="var(--surface)" strokeWidth={2.5} />
    </g>
  );
}

function Txt({ x, y, children, color, size = 22, anchor = 'start', weight = 700 }: { x: number; y: number; children: ReactNode; color: string; size?: number; anchor?: 'start' | 'middle' | 'end'; weight?: number }) {
  return (
    <text x={x} y={y} fontSize={size} fontWeight={weight} fill={color} textAnchor={anchor} stroke="var(--surface)" strokeWidth={5} paintOrder="stroke" strokeLinejoin="round">
      {children}
    </text>
  );
}

/** Lesson 2's step triangle: one step right (+1), then up by the slope. */
function StepTriangle({ F, x0, show }: { F: Frame; x0: number; show: boolean }) {
  const t = stepTriangle(x0);
  return (
    <g style={{ opacity: show ? 1 : 0, transition: 'opacity 0.45s' }}>
      <line x1={F.sx(t.x0)} y1={F.sy(t.y0)} x2={F.sx(t.x1)} y2={F.sy(t.y0)} stroke="var(--ink-2)" strokeWidth={3} strokeDasharray="7 5" />
      <line x1={F.sx(t.x1)} y1={F.sy(t.y0)} x2={F.sx(t.x1)} y2={F.sy(t.y1)} stroke={C.slope} strokeWidth={4.5} />
      <Txt x={(F.sx(t.x0) + F.sx(t.x1)) / 2} y={F.sy(t.y0) + 26} color="var(--ink-2)" anchor="middle" size={20}>
        +1
      </Txt>
      <Txt x={F.sx(t.x1) + 10} y={(F.sy(t.y0) + F.sy(t.y1)) / 2 + 8} color={C.slope} size={22}>
        +{fmt(t.rise)}
      </Txt>
    </g>
  );
}

/** A labelled step triangle on the curve itself: run along w, then the change in L. */
function CurveTriangle({ F, w0, w1, color = C.slope }: { F: Frame; w0: number; w1: number; color?: string }) {
  const L0 = loss(w0);
  const L1 = loss(w1);
  return (
    <g>
      <line x1={F.sx(w0)} y1={F.sy(L0)} x2={F.sx(w1)} y2={F.sy(L0)} stroke="var(--ink-2)" strokeWidth={3} strokeDasharray="7 5" />
      <line x1={F.sx(w1)} y1={F.sy(L0)} x2={F.sx(w1)} y2={F.sy(L1)} stroke={color} strokeWidth={4.5} />
    </g>
  );
}

// ---- The scene -----------------------------------------------------------------------------------

function Scene({ step }: { step: number }) {
  const { tr } = useT();
  const [s, set] = useLessonState(ID, { w: W0 });
  const [touched, setTouched] = useState(false);
  useEffect(() => setTouched(false), [step]);

  // Automatic sequences, one per step that has one.
  const turn = useTicker(step === 0, 1, 1200);
  const SWEEP = 150;
  const sweep = useTicker(step === 1, SWEEP, 26);
  const tri = useTicker(step === 2, LINE_TRIANGLES.length, 650);
  const tan = useTicker(step === 3, TANGENT_AT.length, 650);
  const zoomTick = useTicker(step === 4, 2, 1900);
  const nudgeTick = useTicker(step === 5, 4, 1400);
  const demo = useTicker(step === 7 && !touched, SIGN_DEMO.length - 1, 1900);

  // The camera: zoom level z (×10^z) and the blend towards lesson 2's line (step 3).
  const zTarget = step === 4 ? Z_STEPS[Math.min(zoomTick + 1, 2)] : step === 5 ? Z_NUDGE : 0;
  const z = useTween(zTarget, 1700);
  const lineMix = useTween(step === 2 ? 1 : 0, 900);
  const F = frameOf(mixView(zoomView(z), LINE_VIEW, lineMix));

  // The knob w (the dot on the curve, the knob and the needle all show this one value).
  const sweepW = W_MIN + ((W_MAX - W_MIN) * sweep) / SWEEP;
  const live = step === 1 || (step === 7 && touched);
  const wTarget =
    step === 0 ? (turn ? W_TURN : W0) : step === 1 ? sweepW : step === 5 ? (nudgeTick >= 1 ? NUDGED.w1 : W0) : step === 7 ? (touched ? s.w : SIGN_DEMO[demo]) : W0;
  const w = useTween(wTarget, live ? 0 : 900);
  const L = loss(w);
  const m = slope(w);

  const uOpacity = 1 - lineMix;
  const showCurve = step >= 1;
  const curveTo = step === 1 ? sweepW : W_MAX;
  const sweepDone = step === 1 && sweep >= SWEEP;

  const onPointer = (e: React.PointerEvent<SVGSVGElement>, start: boolean) => {
    if (step !== 7) return;
    if (!start && e.buttons === 0) return;
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * PW;
    const x = F.v.x0 + (px - PAD.l) / F.kx;
    const snapped = Math.round(Math.max(0.5, Math.min(9.5, x)) * 20) / 20;
    if (start) e.currentTarget.setPointerCapture(e.pointerId);
    setTouched(true);
    set({ w: snapped });
  };

  const zoomLabel = `×${fmt(10 ** z, 0)}`;
  const sign = Math.abs(m) < 1e-9 ? 0 : Math.sign(m);
  const signNote =
    sign < 0
      ? bi('negative: increase w, the loss goes down', 'منفی: w را زیاد کن، خطا کم می‌شود')
      : sign > 0
        ? bi('positive: increase w, the loss goes up', 'مثبت: w را زیاد کن، خطا زیاد می‌شود')
        : bi('zero: the bottom of the valley', 'صفر: ته دره');

  return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 18, minHeight: 0 }}>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 22, flex: 'none', width: 170 }}>
        <LiveKnob value={w} digits={step === 5 ? 3 : 2} dim={step === 2} />
        <div style={{ zoom: 1.12 }}>
          <LiveMeter value={L} digits={step === 5 ? 6 : 2} dim={step === 2} />
        </div>
      </div>

      <div style={{ position: 'relative', flex: '1 1 auto', maxWidth: PW, minWidth: 0 }}>
        <svg
          viewBox={`0 0 ${PW} ${PH}`}
          width="100%"
          style={{ display: 'block', touchAction: step === 7 ? 'none' : undefined, cursor: step === 7 ? 'grab' : undefined, userSelect: 'none' }}
          role="img"
          aria-label="L(w)"
          direction="ltr"
          onPointerDown={(e) => onPointer(e, true)}
          onPointerMove={(e) => onPointer(e, false)}
          data-testid="derivative-plot"
        >
          <defs>
            <clipPath id="d5-clip">
              <rect x={PAD.l} y={PAD.t} width={PW - PAD.l - PAD.r} height={PH - PAD.t - PAD.b} />
            </clipPath>
          </defs>
          <Axes
            F={F}
            xName={lineMix > 0.5 ? 'x' : 'w'}
            yName={lineMix > 0.5 ? 'ŷ' : 'L'}
            xColor={lineMix > 0.5 ? C.input : C.weight}
            yColor={lineMix > 0.5 ? C.output : C.loss}
          />
          <g clipPath="url(#d5-clip)">
            {/* lesson 2's line and its step triangles (step 3) */}
            <g opacity={lineMix}>
              <path d={fnPath(F, line, -1, 5, 4)} fill="none" stroke={C.output} strokeWidth={4} />
              {LINE_TRIANGLES.map((x0, i) => (
                <StepTriangle key={x0} F={F} x0={x0} show={step === 2 && tri > i} />
              ))}
            </g>

            <g opacity={uOpacity}>
              {/* the U curve L(w) */}
              {showCurve && <path d={fnPath(F, loss, W_MIN, curveTo)} fill="none" stroke={C.loss} strokeWidth={4.5} strokeLinejoin="round" opacity={step === 3 ? 0.35 : 1} style={{ transition: 'opacity 0.5s' }} />}

              {/* step 1: two readings */}
              {step === 0 && (
                <g>
                  <Dot F={F} x={W0} y={loss(W0)} color={C.weight} r={8} />
                  <Txt x={F.sx(W0) + 14} y={F.sy(loss(W0)) - 12} color="var(--ink-2)" size={20}>
                    ({fmt(W0)}, {fmt(loss(W0))})
                  </Txt>
                  <g style={{ opacity: turn ? 1 : 0, transition: 'opacity 0.5s 0.5s' }}>
                    <Dot F={F} x={W_TURN} y={loss(W_TURN)} color={C.weight} r={8} />
                    <Txt x={F.sx(W_TURN) + 14} y={F.sy(loss(W_TURN)) + 26} color="var(--ink-2)" size={20}>
                      ({fmt(W_TURN)}, {fmt(loss(W_TURN))})
                    </Txt>
                  </g>
                </g>
              )}

              {/* step 2: the bottom of the curve */}
              <g style={{ opacity: sweepDone || step === 3 ? 1 : 0, transition: 'opacity 0.5s' }}>
                <circle cx={F.sx(W_BEST)} cy={F.sy(loss(W_BEST))} r={8} fill={C.output} stroke="var(--surface)" strokeWidth={2.5} />
                {step === 1 && (
                  <>
                    <Txt x={F.sx(W_BEST)} y={F.sy(loss(W_BEST)) - 70} color={C.output} anchor="middle" size={22}>
                      {tr(bi('best', 'بهترین'))}
                    </Txt>
                    <Txt x={F.sx(W_BEST)} y={F.sy(loss(W_BEST)) - 42} color={C.weight} anchor="middle" size={22}>
                      w = {fmt(W_BEST)}
                    </Txt>
                  </>
                )}
              </g>

              {/* step 4: tangent pieces where the curve is steep, gentle, flat */}
              {step === 3 &&
                TANGENT_AT.map((x, i) => {
                  const e = tangentEnds(F, x, loss(x), slope(x), 62);
                  const words = [bi('steep', 'تند'), bi('gentle', 'ملایم'), bi('flat', 'صاف')];
                  return (
                    <g key={x} style={{ opacity: tan > i ? 1 : 0, transition: 'opacity 0.4s' }}>
                      <line {...e} stroke={C.slope} strokeWidth={5} strokeLinecap="round" />
                      <Dot F={F} x={x} y={loss(x)} color={C.slope} r={7} />
                      {i < words.length && (
                        <Txt x={F.sx(x) + (i === 2 ? 0 : 22)} y={F.sy(loss(x)) + (i === 2 ? -22 : -14)} color={C.slope} anchor={i === 2 ? 'middle' : 'start'} size={22}>
                          {tr(words[i])}
                        </Txt>
                      )}
                    </g>
                  );
                })}

              {/* step 5: the step triangle on the zoomed curve */}
              {step === 4 && (
                <g style={{ opacity: zoomTick >= 2 ? 1 : 0, transition: 'opacity 0.5s' }}>
                  <CurveTriangle F={F} w0={W0} w1={W0 + PEEK_RUN} />
                  <Txt x={F.sx(W0 + PEEK_RUN) + 14} y={(F.sy(loss(W0)) + F.sy(loss(W0 + PEEK_RUN))) / 2 + 8} color={C.slope} size={24}>
                    {tr(bi('slope = ?', 'شیب = ?'))}
                  </Txt>
                </g>
              )}

              {/* step 6: the nudge, with real numbers */}
              {step === 5 && (
                <g>
                  <g style={{ opacity: nudgeTick >= 1 ? 1 : 0, transition: 'opacity 0.5s' }}>
                    <line x1={F.sx(NUDGED.w0)} y1={F.sy(NUDGED.L0)} x2={F.sx(NUDGED.w1)} y2={F.sy(NUDGED.L0)} stroke="var(--ink-2)" strokeWidth={3} strokeDasharray="7 5" />
                    <Txt x={(F.sx(NUDGED.w0) + F.sx(NUDGED.w1)) / 2} y={F.sy(NUDGED.L0) - 14} color={C.weight} anchor="middle" size={21}>
                      Δw = {fmt(NUDGED.dw, 3)}
                    </Txt>
                    <Dot F={F} x={NUDGED.w1} y={NUDGED.L1} color={C.weight} r={8} />
                    <Txt x={F.sx(NUDGED.w1) - 16} y={F.sy(NUDGED.L1) + 30} color="var(--ink-2)" anchor="end" size={20}>
                      ({fmt(NUDGED.w1, 3)}, {fmt(NUDGED.L1, 6)})
                    </Txt>
                  </g>
                  <g style={{ opacity: nudgeTick >= 2 ? 1 : 0, transition: 'opacity 0.5s' }}>
                    <line x1={F.sx(NUDGED.w1)} y1={F.sy(NUDGED.L0)} x2={F.sx(NUDGED.w1)} y2={F.sy(NUDGED.L1)} stroke={C.loss} strokeWidth={4.5} />
                    <Txt x={F.sx(NUDGED.w1) + 12} y={(F.sy(NUDGED.L0) + F.sy(NUDGED.L1)) / 2 + 8} color={C.loss} size={21}>
                      ΔL = {fmt(NUDGED.dL, 6)}
                    </Txt>
                  </g>
                  <Dot F={F} x={NUDGED.w0} y={NUDGED.L0} color={C.weight} r={8} />
                  <Txt x={F.sx(NUDGED.w0) - 12} y={F.sy(NUDGED.L0) + 30} color="var(--ink-2)" anchor="end" size={20}>
                    ({fmt(NUDGED.w0)}, {fmt(NUDGED.L0)})
                  </Txt>
                </g>
              )}

              {/* steps 7–9: the tangent line at the point */}
              {step >= 6 && <path d={fnPath(F, (x) => L + m * (x - w), F.v.x0, F.v.x1, 2)} fill="none" stroke={C.slope} strokeWidth={step === 8 ? 2.5 : 4} opacity={step === 8 ? 0.5 : 1} />}

              {/* the moving point */}
              {(step === 1 || step === 3 || step === 4 || step >= 6) && <Dot F={F} x={w} y={L} color={C.weight} r={step === 7 ? 10 : 9} halo={step === 7} />}

              {/* step 8: the sign badge */}
              {step === 7 && (
                <g transform={`translate(${F.sx(w)}, ${F.sy(L) - 46})`}>
                  <circle r={21} fill={C.slope} />
                  <text y={9} textAnchor="middle" fontSize={28} fontWeight={800} fill="var(--surface)">
                    {sign < 0 ? '−' : sign > 0 ? '+' : '0'}
                  </text>
                </g>
              )}

              {/* step 9: the compass */}
              {step === 8 && (
                <g transform={`translate(${F.sx(w)}, ${F.sy(L)})`}>
                  <circle r={54} fill="var(--surface)" fillOpacity={0.85} stroke="var(--ink-2)" strokeWidth={2.5} />
                  {[0, 90, 180, 270].map((a) => (
                    <line key={a} x1={0} y1={-54} x2={0} y2={-44} stroke="var(--ink-2)" strokeWidth={2.5} transform={`rotate(${a})`} />
                  ))}
                  <g transform={`rotate(${sign < 0 ? 0 : sign > 0 ? 180 : 0})`}>
                    <path d="M0,-8 L-40,0 L0,8 Z" fill="var(--line-strong)" />
                    <path d="M0,-9 L30,-9 L50,0 L30,9 L0,9 Z" fill={C.slope} />
                    <circle r={5} fill="var(--ink)" />
                  </g>
                </g>
              )}
            </g>
          </g>

          {/* zoom badge */}
          <g style={{ opacity: step === 4 || step === 5 || (step === 6 && z > 0.02) ? 1 : 0, transition: 'opacity 0.4s' }}>
            <rect x={PAD.l + 12} y={PAD.t + 12} width={120} height={42} rx={8} fill="var(--surface)" stroke="var(--line-strong)" />
            <text x={PAD.l + 72} y={PAD.t + 41} textAnchor="middle" fontSize={24} fontWeight={700} fill="var(--ink)">
              {zoomLabel}
            </text>
          </g>
          {step === 2 && (
            <Txt x={PAD.l + 18} y={PAD.t + 36} color={C.slope} size={26}>
              {tr(bi('slope', 'شیب'))} = {fmt(LINE_W)}
            </Txt>
          )}
        </svg>

        {/* step 6: change in loss ÷ nudge */}
        {step === 5 && (
          <div style={{ position: 'absolute', top: '5.5%', right: '3.5%', fontSize: '1.25em' }} dir="ltr">
            <Appear show={nudgeTick >= 3} keep>
              <div className="vcard" style={{ borderColor: C.slope }}>
                <Tex display>{`\\frac{\\cL{\\Delta L}}{\\cw{\\Delta w}} = \\frac{\\cL{${tex(NUDGED.dL, 6)}}}{\\cw{${tex(NUDGED.dw, 3)}}} = \\cg{${tex(NUDGED.rate, 3)}}`}</Tex>
                <Appear show={nudgeTick >= 4} keep>
                  <div className="num" style={{ color: C.slope, fontWeight: 700, fontSize: '1.3em', textAlign: 'center' }}>
                    ≈ {fmt(Math.round(NUDGED.rate))}
                  </div>
                </Appear>
              </div>
            </Appear>
          </div>
        )}

        {/* steps 7–9: the derivative card */}
        {step >= 6 && (
          <div style={{ position: 'absolute', top: '5%', left: '50%', transform: 'translateX(-50%)', zoom: 1.15 }}>
            <ValueCard
              label={<Tex>{'\\dfrac{dL}{dw}'}</Tex>}
              value={fmt(m, 2)}
              color={C.slope}
              note={step === 6 ? tr(bi('= the slope of the pink line', '= شیب خط صورتی')) : step === 8 ? tr(bi('go right: increase w', 'برو به راست: w را زیاد کن')) : tr(signNote)}
            />
          </div>
        )}
      </div>
    </div>
  );
}

export const derivativeLesson: StoryLesson = {
  id: ID,
  Scene,
  steps: [
    {
      say: bi('Turn the knob w a little and watch the loss: w from 3 to 3.5, loss from 16 to 9.', 'پیچ w را کمی بچرخان و خطا را نگاه کن: w از ۳ به ۳٫۵، خطا از ۱۶ به ۹.'),
      more: bi(
        'With w = 3 the machine says 2 × 3 = 6, and the loss is $(6 - 10)^2 = 16$. With w = 3.5 it says 7, and the loss is $(7 - 10)^2 = 9$. Each reading is a pair (w, loss), so it can be a dot, just like the dots of lesson 1.',
        'با w = ۳ ماشین می‌گوید ۲ × ۳ = ۶ و خطا می‌شود $(6 - 10)^2 = 16$. با w = ۳٫۵ می‌گوید ۷ و خطا می‌شود $(7 - 10)^2 = 9$. هر بار یک جفت (w، خطا) داریم، پس می‌شود آن را یک نقطه کرد، درست مثل نقطه‌های درس ۱.',
      ),
    },
    {
      say: bi(
        'Compute the loss for every w and plot it: a U-shaped curve. The bottom of the curve is the best w.',
        'برای هر w خطا را حساب کن و روی نمودار بگذار: یک منحنی U شکل. ته منحنی بهترین w است.',
      ),
      more: bi(
        'The loss for any w is $L = (2w - 10)^2$: the guess for 2 notebooks minus 10, squared. Far to the left or right the loss is big; in the middle it drops to 0. At w = 5 the machine says exactly 10, so that is the knob we are looking for.',
        'خطا برای هر w می‌شود $L = (2w - 10)^2$: حدس برای ۲ دفترچه منهای ۱۰، به توان دو. خیلی چپ یا خیلی راست، خطا بزرگ است؛ وسط تا ۰ پایین می‌آید. در w = ۵ ماشین دقیقاً ۱۰ می‌گوید، پس پیچی که دنبالش هستیم همین است.',
      ),
    },
    {
      say: bi(
        'Remember the steepness of the line in lesson 2, the slope? On a straight line the slope is the same everywhere.',
        'یادت هست در درس ۲ به تندی خط گفتیم شیب؟ روی خط راست، شیب همه‌جا یکی است.',
      ),
      recall: { lesson: 'neuron', note: bi('slope = how much it goes up for one step right', 'شیب = با یک قدم به راست، چقدر بالا می‌رود') },
      more: bi(
        'This is the line of the price machine with w = 3. Wherever you put the triangle, one step to the right (+1) always goes up by the same amount (+3). That amount is the slope, and on a straight line it never changes.',
        'این خط ماشین قیمت با w = ۳ است. مثلث را هر جا بگذاری، یک قدم به راست (۱+) همیشه به یک اندازه بالا می‌رود (۳+). این اندازه همان شیب است و روی خط راست هیچ‌وقت عوض نمی‌شود.',
      ),
    },
    {
      say: bi(
        "But a curve isn't equally steep everywhere: steep here, gentle there. So we ask for the slope at a point.",
        'ولی منحنی همه‌جا یک‌جور تند نیست: این‌جا تند، آن‌جا آرام. پس شیب را باید «در یک نقطه» پرسید.',
      ),
      more: bi(
        'Each pink piece shows which way the curve heads at that spot. Near w = 1 it falls fast, near w = 3 more gently, at w = 5 it is flat, and at w = 7 it rises. A single number for "the slope of the curve" makes no sense; it depends on where you stand.',
        'هر تکه صورتی نشان می‌دهد منحنی در آن نقطه به کدام طرف می‌رود. نزدیک w = ۱ تند پایین می‌آید، نزدیک w = ۳ آرام‌تر، در w = ۵ صاف است و در w = ۷ بالا می‌رود. یک عدد برای «شیب کل منحنی» معنی ندارد؛ بستگی دارد کجا ایستاده‌ای.',
      ),
    },
    {
      say: bi(
        "Zoom in at w = 3… and again… Up close the curve looks like a straight line. The slope of that line is the curve's slope at that point.",
        'روی w = ۳ زوم کن... باز هم... از نزدیک، منحنی یک خط راست به نظر می‌رسد. شیبِ این خط، شیب منحنی در همان نقطه است.',
      ),
      more: bi(
        'Watch the numbers on the axes: each zoom shows a piece 10 times smaller. The bend of the curve shrinks much faster than the piece, so at ×100 you can no longer see it. On a straight piece we know what to do: draw a step triangle and read its slope.',
        'به عددهای کنار محورها نگاه کن: هر زوم تکه‌ای ۱۰ برابر کوچک‌تر را نشان می‌دهد. خمیدگی منحنی خیلی سریع‌تر از خود تکه کوچک می‌شود، برای همین در ×۱۰۰ دیگر دیده نمی‌شود. روی یک تکه صاف بلدیم چه کنیم: یک مثلث پله‌ای بکش و شیبش را بخوان.',
      ),
    },
    {
      say: bi(
        "It's easy to compute: give w a tiny nudge. w 3 → 3.001, loss 16 → 15.984. Change in loss ÷ nudge ≈ −16.",
        'حسابش ساده است: یک تلنگر کوچک به w بزن. w از ۳ به ۳٫۰۰۱، خطا از ۱۶ به ۱۵٫۹۸۴. تغییر خطا ÷ تلنگر ≈ −۱۶.',
      ),
      more: bi(
        'Δ (delta) just means "change in". $\\Delta w = 0.001$ is the nudge, and $\\Delta L = 15.984004 - 16 = -0.015996$ is how much the loss changed. Divide: $-0.015996 \\div 0.001 = -15.996$, almost exactly $-16$. It is the same step triangle as in lesson 2, only very small.',
        'Δ (دلتا) فقط یعنی «تغییرِ». $\\Delta w = 0.001$ همان تلنگر است و $\\Delta L = 15.984004 - 16 = -0.015996$ مقدار تغییر خطاست. تقسیم کن: $-0.015996 \\div 0.001 = -15.996$، تقریباً دقیقاً $-16$. همان مثلث پله‌ای درس ۲ است، فقط خیلی کوچک.',
      ),
    },
    {
      say: bi(
        'This number is the derivative, $dL/dw$: "for a tiny nudge to w, how many times as much does the loss change?" Slope and derivative are the same thing.',
        'به این عدد می‌گوییم مشتق، $dL/dw$: «با یک تلنگر کوچک به w، خطا چند برابر تغییر می‌کند؟» شیب و مشتق یک چیزند.',
      ),
      more: bi(
        '$dL/dw$ is read "d L by d w": the change in L divided by a tiny change in w. Here it is $-16$: the loss changes 16 times as much as the nudge, in the opposite direction. The pink line is the straight line we saw when zoomed in; it touches the curve at w = 3.',
        '$dL/dw$ را می‌خوانیم «دی اِل بر دی دبلیو»: تغییر L تقسیم بر یک تغییر خیلی کوچک w. این‌جا $-16$ است: خطا ۱۶ برابرِ تلنگر تغییر می‌کند، در جهت مخالف. خط صورتی همان خط راستی است که در زوم دیدیم؛ در w = ۳ به منحنی می‌چسبد.',
      ),
    },
    {
      say: bi(
        "Its sign tells you which way to go: negative means increasing w lowers the loss; positive the opposite; zero means you're at the bottom.",
        'علامتش راه را نشان می‌دهد: منفی یعنی w را زیاد کنی خطا کم می‌شود؛ مثبت برعکس؛ صفر یعنی ته دره‌ای.',
      ),
      more: bi(
        'Drag the point along the curve. On the left side the line slopes down (negative), so moving right helps. On the right side it slopes up (positive), so moving left helps. At w = 5 it is flat (zero): no direction makes the loss smaller.',
        'نقطه را روی منحنی بکش. سمت چپ خط رو به پایین است (منفی)، پس رفتن به راست کمک می‌کند. سمت راست رو به بالاست (مثبت)، پس رفتن به چپ کمک می‌کند. در w = ۵ صاف است (صفر): هیچ جهتی خطا را کمتر نمی‌کند.',
      ),
    },
    {
      say: bi('So the derivative is learning\'s compass. Now we just need to take steps.', 'پس مشتق قطب‌نمای یادگیری است. حالا فقط باید قدم برداریم.'),
      more: bi(
        'At w = 3 the derivative is $-16$, so the needle points to the right: increase w. It also says how steep the hill is here. The next lesson uses exactly this number to decide each step.',
        'در w = ۳ مشتق $-16$ است، پس عقربه به راست اشاره می‌کند: w را زیاد کن. همچنین می‌گوید تپه این‌جا چقدر تند است. درس بعد دقیقاً با همین عدد اندازه و جهت هر قدم را تعیین می‌کند.',
      ),
    },
  ],
};
