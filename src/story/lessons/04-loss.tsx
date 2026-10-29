// Lesson 4 · How wrong is it? The loss. One number for "how bad"; it is not the output.
// Script: docs/storyboard.md, lesson 4.

import { bi, useT } from '../../i18n';
import { fmt, tex } from '../../engine';
import type { StoryLesson } from '../types';
import { Appear, C, Formula, Knob, Machine, Meter, Note, Stage, ValueCard } from '../kit';
import { useTween, useTweenArray, useTicker } from '../useTween';
import { COMPARE, X, W0, Y, errorOf, guess, lossOf, meanLoss, mirrorGuess, pourShape, receiptRows } from './math-loss';

const ID = 'loss';

const YHAT = guess(W0, X);
const ERR = errorOf(YHAT, Y);
const LOSS = lossOf(YHAT, Y);
const MIRROR = mirrorGuess(YHAT, Y);
const SIZE = Math.abs(ERR);
/** The loss meter's full scale; the same in lesson 5 (the knob w + loss meter pair). */
const METER_MAX = 100;

// ---- The number line with the error and its square (steps 1–5) ---------------------------

const NL = { w: 720, h: 500, left: 40, right: 690, y: 132, d0: 0, d1: 16, sqTop: 168 };
const sx = (v: number) => NL.left + ((v - NL.d0) / (NL.d1 - NL.d0)) * (NL.right - NL.left);
const U = sx(1) - sx(0);
const TICKS = Array.from({ length: NL.d1 - NL.d0 + 1 }, (_, i) => NL.d0 + i);

/** Order the comparison squares appear in on step 4: the real error first, then the others. */
const REVEAL = [SIZE, ...COMPARE.filter((c) => c !== SIZE)];

function Arrow({ from, to, y, color, label, dash }: { from: number; to: number; y: number; color: string; label: string; dash?: string }) {
  const x0 = sx(from);
  const x1 = sx(to);
  const dir = Math.sign(x1 - x0);
  return (
    <g>
      <line x1={x0} y1={y} x2={x1 - dir * 12} y2={y} stroke={color} strokeWidth={4} strokeDasharray={dash} />
      <path d={`M${x1},${y} l${-dir * 16},-9 l0,18 z`} fill={color} />
      <text x={(x0 + x1) / 2} y={y - 12} textAnchor="middle" fontSize={26} fontWeight={700} fill={color}>
        {label}
      </text>
    </g>
  );
}

function ErrorSquare({ side, size, label, labelRight, opacity }: { side: number; size: number; label: string; labelRight: boolean; opacity: number }) {
  if (side < 0.02) return null;
  const x0 = sx(Y - side);
  const px = side * U;
  const cells = Array.from({ length: Math.ceil(side) - 1 }, (_, k) => k + 1);
  return (
    <g>
      <rect x={x0} y={NL.sqTop} width={px} height={px} fill={C.loss} fillOpacity={opacity} stroke={C.loss} strokeWidth={2.5} />
      {cells.map((k) => (
        <g key={k} stroke={C.loss} strokeOpacity={0.35} strokeWidth={1}>
          <line x1={sx(Y - k)} x2={sx(Y - k)} y1={NL.sqTop} y2={NL.sqTop + px} />
          <line x1={x0} x2={x0 + px} y1={NL.sqTop + k * U} y2={NL.sqTop + k * U} />
        </g>
      ))}
      {side > size * 0.9 &&
        (labelRight ? (
          <text x={sx(Y) + 12} y={NL.sqTop + U / 2 + 8} fontSize={22} fontWeight={700} fill={C.loss}>
            {label}
          </text>
        ) : (
          <text x={x0 + 12} y={NL.sqTop + px - 14} fontSize={24} fontWeight={700} fill={C.loss}>
            {label}
          </text>
        ))}
    </g>
  );
}

function NumberLineSvg({ step, tick }: { step: number; tick: number }) {
  const { tr } = useT();
  const shown = (c: number) => (step === 3 && REVEAL.indexOf(c) <= tick) || (step === 4 && c === SIZE);
  const sides = useTweenArray(
    COMPARE.map((c) => (shown(c) ? c : 0)),
    800,
  );
  const mirror = useTween(step === 2 ? 1 : 0, 500);
  const named = step >= 1;
  const labelY = 84;
  const wordY = 44;
  return (
    <svg viewBox={`0 0 ${NL.w} ${NL.h}`} width="100%" style={{ maxWidth: NL.w, maxHeight: '100%', display: 'block' }} role="img" aria-label="number line" direction="ltr">
      {/* squares, largest at the back */}
      {[...COMPARE.keys()]
        .sort((a, b) => COMPARE[b] - COMPARE[a])
        .map((i) => {
          const c = COMPARE[i];
          const area = lossOf(Y - c, Y);
          const label = step === 4 ? `L = ${fmt(area)}` : `${c}² = ${fmt(area)}`;
          return <ErrorSquare key={c} side={sides[i]} size={c} label={label} labelRight={c * U < 90} opacity={c === SIZE ? 0.2 : 0.08} />;
        })}
      {/* comparison guesses (hollow) */}
      {COMPARE.map((c, i) =>
        c !== SIZE && sides[i] > 0.02 ? <circle key={c} cx={sx(Y - c)} cy={NL.y} r={9} fill="var(--surface)" stroke={C.output} strokeWidth={3} opacity={Math.min(1, sides[i] / c)} /> : null,
      )}
      {/* the line */}
      <line x1={sx(NL.d0) - 14} x2={sx(NL.d1) + 14} y1={NL.y} y2={NL.y} stroke="var(--ink-2)" strokeWidth={2.5} />
      {TICKS.map((t) => (
        <g key={t}>
          <line x1={sx(t)} x2={sx(t)} y1={NL.y - (t % 2 ? 5 : 8)} y2={NL.y + (t % 2 ? 5 : 8)} stroke="var(--ink-2)" strokeWidth={t % 2 ? 1 : 1.6} />
          {t % 2 === 0 && (
            <text x={sx(t)} y={NL.y + 28} textAnchor="middle" fontSize={16} fill="var(--muted)">
              {t}
            </text>
          )}
        </g>
      ))}
      {/* the error arrow, and its mirror */}
      <g style={{ opacity: named ? 1 : 0, transition: 'opacity 0.5s' }}>
        <Arrow from={Y} to={YHAT} y={NL.y - 26} color={C.loss} label={fmt(ERR)} />
      </g>
      <g opacity={mirror}>
        <Arrow from={Y} to={MIRROR} y={NL.y - 26} color={C.loss} label={`+${fmt(errorOf(MIRROR, Y))}`} dash="8 6" />
        <circle cx={sx(MIRROR)} cy={NL.y} r={11} fill="var(--surface)" stroke={C.output} strokeWidth={3} />
        <text x={sx(MIRROR)} y={labelY} textAnchor="middle" fontSize={30} fontWeight={700} fill={C.output} opacity={0.75}>
          {named ? `ŷ = ${fmt(MIRROR)}` : fmt(MIRROR)}
        </text>
      </g>
      {/* guess and real price */}
      <circle cx={sx(YHAT)} cy={NL.y} r={11} fill={C.output} stroke="var(--surface)" strokeWidth={2} />
      <circle cx={sx(Y)} cy={NL.y} r={11} fill={C.target} stroke="var(--surface)" strokeWidth={2} />
      <text x={sx(YHAT)} y={wordY} textAnchor="middle" fontSize={18} fill={C.output}>
        {tr(bi('guess', 'حدس'))}
      </text>
      <text x={sx(Y)} y={wordY} textAnchor="middle" fontSize={18} fill={C.target}>
        {tr(bi('real', 'واقعی'))}
      </text>
      <text x={sx(YHAT)} y={labelY} textAnchor="middle" fontSize={30} fontWeight={700} fill={C.output}>
        {named ? `ŷ = ${fmt(YHAT)}` : fmt(YHAT)}
      </text>
      <text x={sx(Y)} y={labelY} textAnchor="middle" fontSize={30} fontWeight={700} fill={C.target}>
        {named ? `y = ${fmt(Y)}` : fmt(Y)}
      </text>
    </svg>
  );
}

/** The price machine from lesson 2, small: 2 notebooks, knob w = 3, guess 6. */
function MiniMachine() {
  const { tr } = useT();
  return (
    <div style={{ zoom: 0.62 }}>
      <Machine
        input={X}
        output={fmt(YHAT)}
        rule="\cyhat{\hat y} = \cw{w}\cdot\cx{x}"
        knobs={<Knob label="w" value={W0} min={0} max={10} size={76} />}
        inputLabel={tr(bi('notebooks', 'دفترچه'))}
        outputLabel={tr(bi('guess', 'حدس'))}
      />
    </div>
  );
}

function SidePanel({ step, tick }: { step: number; tick: number }) {
  const { tr } = useT();
  const c2 = Math.max(...COMPARE);
  return (
    <div style={{ width: 340, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14 }}>
      <MiniMachine />
      {step === 0 && (
        <div className="vcard" style={{ borderColor: C.target }}>
          <div className="vcard-label" style={{ color: C.target }}>
            {tr(bi('real price', 'قیمت واقعی'))}
          </div>
          <div className="vcard-value num" style={{ color: C.target }}>
            {fmt(Y)}
          </div>
        </div>
      )}
      {step === 1 && (
        <>
          <Formula size={1.25}>{`\\cyhat{\\hat y} - \\cy{y} = \\cyhat{${tex(YHAT)}} - \\cy{${tex(Y)}} = \\cL{${tex(ERR)}}`}</Formula>
          <Note>{tr(bi(`${fmt(SIZE)} too few`, `${fmt(SIZE)} تا کم`))}</Note>
        </>
      )}
      {step === 2 && (
        <>
          <Formula size={1.2}>{`\\cyhat{${tex(YHAT)}} - \\cy{${tex(Y)}} = \\cL{${tex(ERR)}}`}</Formula>
          <Formula size={1.2}>{`\\cyhat{${tex(MIRROR)}} - \\cy{${tex(Y)}} = \\cL{+${tex(errorOf(MIRROR, Y))}}`}</Formula>
          <Note>{tr(bi(`the same size: ${fmt(SIZE)}`, `اندازه یکی است: ${fmt(SIZE)}`))}</Note>
        </>
      )}
      {step === 3 && (
        <>
          <Formula size={1.3}>{`(${tex(ERR)})^2 = \\cL{${tex(LOSS)}}`}</Formula>
          <Appear show={tick >= 2} keep>
            <Formula size={1}>{`${c2} = ${tex(c2 / SIZE)} \\times ${SIZE} \\;\\Rightarrow\\; \\cL{${tex(lossOf(Y - c2, Y))}} = ${tex(lossOf(Y - c2, Y) / LOSS)} \\times \\cL{${tex(LOSS)}}`}</Formula>
          </Appear>
        </>
      )}
      {step === 4 && (
        <>
          <div style={{ zoom: 1.25 }}>
            <Meter value={LOSS} max={METER_MAX} label="L" digits={2} />
          </div>
          <Formula size={1.1}>{`\\cL{L} = (\\cyhat{\\hat y} - \\cy{y})^2 = (\\cyhat{${tex(YHAT)}} - \\cy{${tex(Y)}})^2 = \\cL{${tex(LOSS)}}`}</Formula>
        </>
      )}
    </div>
  );
}

function LineScene({ step }: { step: number }) {
  const tick = useTicker(step === 3, REVEAL.length - 1, 1500);
  return (
    <Stage row>
      <div style={{ flex: '1 1 600px', maxWidth: NL.w, minWidth: 0, display: 'flex', justifyContent: 'center' }}>
        <NumberLineSvg step={step} tick={tick} />
      </div>
      <SidePanel step={step} tick={tick} />
    </Stage>
  );
}

// ---- Four receipts: each with its square, poured together and levelled (step 6) ----------

const RC = { w: 1000, h: 430, u: 30, base: 410, cw: 7, top: 8, cardW: 196, cardH: 112 };

function ReceiptScene() {
  const { tr } = useT();
  const rows = receiptRows(W0);
  const mean = meanLoss(W0);
  const phase = useTicker(true, 2, 1700);
  const [morph, level] = useTweenArray([phase >= 1 ? 1 : 0, phase >= 2 ? 1 : 0], 1000);
  const cx = (i: number) => RC.w / 8 + (i * RC.w) / 4;
  const sumTex = rows.map((r) => tex(r.loss)).join(' + ');
  const meanH = (mean / RC.cw) * RC.u;
  return (
    <Stage>
      <svg viewBox={`0 0 ${RC.w} ${RC.h}`} width="100%" style={{ maxWidth: 960, display: 'block' }} role="img" aria-label="receipts" direction="ltr">
        {rows.map((r, i) => {
          const run = r.x === X && r.y === Y;
          const { width, height } = pourShape(r.loss, RC.cw, mean, morph, level);
          const pw = width * RC.u;
          const ph = height * RC.u;
          const side = Math.sqrt(r.loss);
          const x0 = cx(i) - pw / 2;
          const y0 = RC.base - ph;
          const shownLoss = r.loss + (mean - r.loss) * level * morph;
          return (
            <g key={i}>
              <rect x={cx(i) - RC.cardW / 2} y={RC.top} width={RC.cardW} height={RC.cardH} rx={10} fill="var(--surface-2)" stroke={run ? 'var(--ink-2)' : 'var(--line-strong)'} strokeWidth={run ? 2.5 : 1.5} />
              <text x={cx(i) - 78} y={RC.top + 34} fontSize={21} fontWeight={700} fill={C.input}>
                x = {r.x}
              </text>
              <text x={cx(i) + 78} y={RC.top + 34} fontSize={21} fontWeight={700} fill={C.target} textAnchor="end">
                y = {fmt(r.y)}
              </text>
              <text x={cx(i) - 78} y={RC.top + 66} fontSize={21} fontWeight={700} fill={C.output}>
                ŷ = {fmt(r.yhat)}
              </text>
              <text x={cx(i) - 78} y={RC.top + 98} fontSize={21} fontWeight={700} fill={C.loss}>
                ŷ − y = {fmt(r.err)}
              </text>
              <rect x={x0} y={y0} width={pw} height={ph} fill={C.loss} fillOpacity={0.18} stroke={C.loss} strokeWidth={2.5} />
              <g opacity={1 - morph} stroke={C.loss} strokeOpacity={0.35} strokeWidth={1}>
                {Array.from({ length: Math.max(0, Math.ceil(side) - 1) }, (_, k) => (
                  <g key={k}>
                    <line x1={x0 + ((k + 1) * pw) / side} x2={x0 + ((k + 1) * pw) / side} y1={y0} y2={RC.base} />
                    <line x1={x0} x2={x0 + pw} y1={y0 + ((k + 1) * ph) / side} y2={y0 + ((k + 1) * ph) / side} />
                  </g>
                ))}
              </g>
              <text x={cx(i)} y={y0 - 10} textAnchor="middle" fontSize={26} fontWeight={700} fill={C.loss}>
                {fmt(shownLoss, 2)}
              </text>
            </g>
          );
        })}
        <line x1={20} x2={RC.w - 20} y1={RC.base} y2={RC.base} stroke="var(--ink-2)" strokeWidth={2} />
        <g opacity={level}>
          <line x1={20} x2={RC.w - 20} y1={RC.base - meanH} y2={RC.base - meanH} stroke={C.loss} strokeWidth={2.5} strokeDasharray="10 7" />
        </g>
      </svg>
      <Appear show={phase >= 2} keep>
        <Formula size={1.35}>{`\\cL{L} = \\frac{${sumTex}}{${rows.length}} = \\cL{${tex(mean)}}`}</Formula>
      </Appear>
      <Note>{tr(bi('each receipt: its own loss → their average: one number', 'هر رسید: خطای خودش ← میانگین‌شان: یک عدد'))}</Note>
    </Stage>
  );
}

// ---- Output vs loss (step 7) and the question (step 8) -------------------------------------

function CardsScene() {
  const { tr } = useT();
  return (
    <Stage>
      <div style={{ display: 'flex', gap: 48, flexWrap: 'wrap', justifyContent: 'center', alignItems: 'flex-start' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
          <div style={{ zoom: 1.3 }}>
            <ValueCard big label={tr(bi('Output ŷ', 'خروجی ŷ'))} value={fmt(YHAT)} color={C.output} note={tr(bi("the machine's answer, in coins", 'جواب ماشین، به سکه'))} />
          </div>
          <Formula size={1.2}>{`\\cyhat{\\hat y} = \\cw{w}\\cdot\\cx{x} = \\cw{${tex(W0)}}\\cdot\\cx{${tex(X)}} = \\cyhat{${tex(YHAT)}}`}</Formula>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
          <div style={{ zoom: 1.3 }}>
            <ValueCard big label={tr(bi('Loss L', 'خطا L'))} value={fmt(LOSS)} color={C.loss} note={tr(bi('how bad it is; 0 = perfect', 'چقدر بد است؛ ۰ = بی‌نقص'))} />
          </div>
          <Formula size={1.2}>{`\\cL{L} = (\\cyhat{${tex(YHAT)}} - \\cy{${tex(Y)}})^2 = \\cL{${tex(LOSS)}}`}</Formula>
        </div>
      </div>
    </Stage>
  );
}

function QuestionScene() {
  const arrow = (ch: string) => (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', color: C.slope, fontWeight: 700, lineHeight: 1 }}>
      <span style={{ fontSize: 40 }}>?</span>
      <span style={{ fontSize: 64 }}>{ch}</span>
    </div>
  );
  return (
    <Stage row>
      <div dir="ltr" style={{ display: 'flex', alignItems: 'center', gap: 26 }}>
        {arrow('←')}
        <div style={{ zoom: 1.5 }}>
          <Knob label="w" value={W0} min={0} max={10} size={110} />
        </div>
        {arrow('→')}
      </div>
      <div style={{ width: 70 }} />
      <div style={{ zoom: 1.6 }}>
        <Meter value={LOSS} max={METER_MAX} label="L" digits={2} />
      </div>
    </Stage>
  );
}

function Scene({ step }: { step: number }) {
  if (step <= 4) return <LineScene step={step} />;
  if (step === 5) return <ReceiptScene />;
  if (step === 6) return <CardsScene />;
  return <QuestionScene />;
}

export const lossLesson: StoryLesson = {
  id: ID,
  Scene,
  steps: [
    {
      say: bi(
        'Back to the simplest machine: one neuron with w = 3. For 2 notebooks it says 6. The real price is 10.',
        'برگردیم سراغ ساده‌ترین ماشین: یک نورون با w = ۳. برای ۲ دفترچه می‌گوید ۶. قیمت واقعی ۱۰ است.',
      ),
      more: bi(
        'This is the price machine from lesson 2: each notebook costs w = 3, so 2 notebooks cost 3 × 2 = 6. But at the shop, 2 notebooks really cost 10 coins. On the number line the green dot (the guess) and the gold dot (the real price) are not in the same place.',
        'این همان ماشین قیمت درس ۲ است: هر دفترچه w = ۳ سکه، پس ۲ دفترچه می‌شود ۳ × ۲ = ۶. ولی در مغازه، ۲ دفترچه واقعاً ۱۰ سکه است. روی محور، نقطه سبز (حدس) و نقطه طلایی (قیمت واقعی) سر یک جا نیستند.',
      ),
    },
    {
      say: bi(
        'We call the right answer y. The error is guess minus answer: 6 − 10 = −4, four too few.',
        'به جواب درست می‌گوییم y. اختلاف = حدس منهای جواب: ۶ − ۱۰ = −۴. یعنی ۴ تا کم.',
      ),
      more: bi(
        'The guess is $\\hat y$ (from lesson 2) and the right answer is $y$. To get from 10 to 6 you walk 4 steps to the left, so the error is $-4$. The minus sign only says "too few".',
        'حدس ماشین $\\hat y$ است (از درس ۲) و جواب درست $y$. از ۱۰ تا ۶، چهار قدم به چپ می‌رویم، پس اختلاف $-4$ است. علامت منفی فقط می‌گوید «کم آورده».',
      ),
    },
    {
      say: bi(
        "Four too few or four too many: equally bad. The sign doesn't matter, the size does.",
        '۴ تا کم یا ۴ تا زیاد، هر دو به یک اندازه بدند. علامت مهم نیست، اندازه مهم است.',
      ),
      more: bi(
        'If the machine had said 14, it would be 4 too many: 14 − 10 = +4. The two arrows have the same length. For the shop, being 4 coins off is just as bad in both directions.',
        'اگر ماشین گفته بود ۱۴، چهار تا زیاد می‌گفت: ۱۴ منهای ۱۰ می‌شود مثبت ۴. دو پیکان دقیقاً هم‌اندازه‌اند. برای مغازه، ۴ سکه اشتباه از هر طرف به یک اندازه بد است.',
      ),
    },
    {
      say: bi(
        'We square the error: (−4)² = 16. Always positive, and big mistakes cost much more.',
        'اختلاف را به توان دو می‌رسانیم: (−۴)² = ۱۶. همیشه مثبت، و اشتباه بزرگ خیلی بیشتر جریمه می‌شود.',
      ),
      more: bi(
        'Squaring means multiplying a number by itself: $(-4) \\times (-4) = 16$. Think of it as a square with side 4: it has 16 small cells. An error of 1 gives 1 cell, but an error of 8 gives 64 cells: twice the error, four times the penalty.',
        'به توان دو رساندن یعنی عدد را در خودش ضرب کنیم: $(-4) \\times (-4) = 16$. مثل یک مربع با ضلع ۴ که ۱۶ خانه کوچک دارد. اختلاف ۱ فقط ۱ خانه می‌شود، ولی اختلاف ۸ می‌شود ۶۴ خانه: اشتباه دو برابر، جریمه چهار برابر.',
      ),
    },
    {
      say: bi(
        'This number is the loss: $L = (\\hat y - y)^2$. One number that says how bad the machine is. Zero means perfect.',
        'به این عدد می‌گوییم خطا (Loss): $L = (\\hat y - y)^2$. یک عدد که می‌گوید ماشین چقدر بد است. صفر یعنی بی‌نقص.',
      ),
      more: bi(
        'L is short for loss. The red meter shows it: right now it reads 16. If the guess were exactly 10, the square would vanish and the meter would read 0. The smaller L, the better the machine.',
        'L از کلمه Loss می‌آید. عقربه قرمز آن را نشان می‌دهد: الان روی ۱۶ است. اگر حدس دقیقاً ۱۰ بود، مربع ناپدید می‌شد و عقربه روی ۰ می‌ایستاد. هرچه L کوچک‌تر، ماشین بهتر.',
      ),
    },
    {
      say: bi(
        "With several examples, we compute each one's loss and take the average: one number for the whole job.",
        'با چند نمونه، خطای هر کدام را حساب می‌کنیم و میانگین می‌گیریم: یک عدد برای کل کار.',
      ),
      more: bi(
        'Here are 4 receipts from the shop; a notebook costs about 5. With w = 3 the machine misses each one by a different amount, so each has its own square. Pour the squares together and spread them evenly: every column gets the average, 27.5.',
        'این‌ها ۴ رسید از مغازه‌اند؛ هر دفترچه حدوداً ۵ سکه است. با w = ۳ ماشین هر کدام را به اندازه‌ای متفاوت اشتباه می‌کند، پس هر کدام مربع خودش را دارد. مربع‌ها را روی هم بریز و یکسان پخش کن: سهم هر ستون می‌شود میانگین، یعنی ۲۷٫۵.',
      ),
    },
    {
      say: bi(
        "Don't mix up two numbers: the output (6) is the machine's answer; the loss (16) is its badness score.",
        'دو عدد را قاطی نکن: خروجی (۶) جوابِ ماشین است؛ خطا (۱۶) نمره بد بودنش.',
      ),
      more: bi(
        'The output $\\hat y$ is a price in coins: what the machine says. The loss $L$ is not a price; it is a score we compute afterwards by comparing the answer with the truth. A machine can give a big output and have a small loss, or the other way round.',
        'خروجی $\\hat y$ یک قیمت است به سکه: چیزی که ماشین می‌گوید. خطا $L$ قیمت نیست؛ نمره‌ای است که بعداً با مقایسه جواب و واقعیت حساب می‌کنیم. ماشین می‌تواند خروجی بزرگ و خطای کوچک داشته باشد، یا برعکس.',
      ),
    },
    {
      say: bi(
        'Learning means turning the knobs so the loss goes down. But which way?',
        'یادگیری یعنی پیچ‌ها را طوری بچرخان که خطا کم شود. ولی به کدام طرف؟',
      ),
      more: bi(
        'The knob w sets the guess, and the guess sets the loss. Turn w one way and the meter may go down; turn it the other way and it may go up. The next lesson finds a way to know the right direction without guessing.',
        'پیچ w حدس را تعیین می‌کند و حدس، خطا را. اگر w را یک طرف بچرخانی شاید عقربه پایین بیاید؛ طرف دیگر شاید بالا برود. درس بعد راهی پیدا می‌کند که جهت درست را بدون حدس زدن بدانیم.',
      ),
    },
  ],
};
