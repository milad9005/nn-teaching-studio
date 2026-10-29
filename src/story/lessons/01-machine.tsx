// Lesson 1 · The number machine. A function is a machine with a rule; its picture is a graph.
// Script: docs/storyboard.md, lesson 1.

import { bi, useT } from '../../i18n';
import { useLessonState } from '../../state/lessonState';
import { Plot2D, type PlotLayer } from '../../components/Plot2D';
import { fmt } from '../../engine';
import type { StoryLesson } from '../types';
import { Appear, C, Machine, Note, Stage } from '../kit';
import { useTicker } from '../useTween';

const ID = 'machine';
const rule = (x: number) => 3 * x;
const TABLE = [0, 1, 2, 3];

function Table({ rows, named }: { rows: number; named: boolean }) {
  const { tr } = useT();
  return (
    <table className="tbl" style={{ maxWidth: 260 }} dir="ltr">
      <thead>
        <tr>
          <th className="num" style={{ color: C.input }}>
            {named ? 'x' : tr(bi('input', 'ورودی'))}
          </th>
          <th className="num" style={{ color: C.output }}>
            {named ? 'f(x)' : tr(bi('output', 'خروجی'))}
          </th>
        </tr>
      </thead>
      <tbody>
        {TABLE.map((x, i) => (
          <tr key={x} style={{ opacity: i < rows ? 1 : 0.15, transition: 'opacity 0.4s' }}>
            <td className="num">{x}</td>
            <td className="num">{i < rows ? rule(x) : ''}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Graph({ dots, line, x, onDrag }: { dots: number; line: boolean; x: number; onDrag?: (x: number) => void }) {
  const { tr } = useT();
  const layers: PlotLayer[] = [];
  if (line) layers.push({ kind: 'fn', f: rule, color: C.output, width: 3 });
  TABLE.slice(0, dots).forEach((v) => {
    layers.push(
      { kind: 'segment', from: [v, 0], to: [v, rule(v)], color: C.input, dash: '4 4', width: 1.2, opacity: 0.6 },
      { kind: 'segment', from: [0, rule(v)], to: [v, rule(v)], color: C.output, dash: '4 4', width: 1.2, opacity: 0.6 },
    );
  });
  layers.push({ kind: 'points', color: C.output, pts: TABLE.slice(0, dots).map((v) => ({ x: v, y: rule(v), r: 6, label: `(${v}, ${rule(v)})` })) });
  if (onDrag) {
    layers.push(
      { kind: 'segment', from: [x, 0], to: [x, rule(x)], color: C.input, dash: '5 4', width: 2 },
      { kind: 'segment', from: [0, rule(x)], to: [x, rule(x)], color: C.output, dash: '5 4', width: 2 },
      { kind: 'points', color: C.input, pts: [{ x, y: rule(x), r: 7, label: `(${fmt(x, 2)}, ${fmt(rule(x), 2)})` }] },
    );
  }
  return (
    <div style={{ width: '100%', maxWidth: 620 }}>
      <Plot2D
        xDomain={[-0.5, 4]}
        yDomain={[-1, 12.5]}
        layers={layers}
        height={330}
        xLabel={tr(bi('input → right', 'ورودی → به راست'))}
        yLabel={tr(bi('output ↑ up', 'خروجی ↑ به بالا'))}
        handle={onDrag ? { x, y: rule(x), color: C.input } : undefined}
        onDrag={onDrag ? (v) => onDrag(Math.round(Math.max(0, Math.min(4, v)) * 10) / 10) : undefined}
        testId="machine-graph"
      />
    </div>
  );
}

/** A small network drawn as circles and lines: the same machine, with a learned rule. */
function Network() {
  const L = [[70], [40, 100, 160], [100]];
  const xs = [60, 200, 340];
  return (
    <svg viewBox="0 0 400 200" width="100%" style={{ maxWidth: 440 }} role="img" aria-label="network">
      {L.slice(0, -1).flatMap((col, l) => col.flatMap((y0) => L[l + 1].map((y1, j) => <line key={`${l}-${y0}-${j}`} x1={xs[l]} y1={y0 + 30} x2={xs[l + 1]} y2={y1} stroke="var(--c-w)" strokeWidth={2} opacity={0.6} />)))}
      {L.map((col, l) => col.map((y, j) => <circle key={`${l}-${j}`} cx={xs[l]} cy={l === 0 ? y + 30 : y} r={18} fill="var(--surface)" stroke={l === 0 ? C.input : l === 2 ? C.output : 'var(--ink-2)'} strokeWidth={3} />))}
      <text x={60} y={105} textAnchor="middle" fontSize={16} fontWeight={700} fill={C.input}>
        2
      </text>
      <text x={340} y={105} textAnchor="middle" fontSize={16} fontWeight={700} fill={C.output}>
        ?
      </text>
    </svg>
  );
}

function Scene({ step }: { step: number }) {
  const { tr } = useT();
  const [s, set] = useLessonState(ID, { x: 2 });
  const cycling = step === 2;
  const tick = useTicker(cycling, TABLE.length - 1, 1100);
  const input = step < 1 ? undefined : step === 2 ? TABLE[tick] : 2;

  if (step <= 3) {
    return (
      <Stage>
        <Machine
          input={input}
          output={input === undefined ? undefined : rule(input)}
          rule={step === 0 ? '?' : step >= 3 ? 'f(x) = 3x' : '\\times 3'}
          pulse={input}
          inputLabel={step >= 3 ? 'x' : tr(bi('input', 'ورودی'))}
          outputLabel={step >= 3 ? 'f(x)' : tr(bi('output', 'خروجی'))}
        />
        <Appear show={step >= 2}>
          <Table rows={step === 2 ? tick + 1 : TABLE.length} named={step >= 3} />
        </Appear>
      </Stage>
    );
  }
  if (step <= 5) {
    return <StepGraph step={step} x={s.x} setX={(x) => set({ x })} />;
  }
  return (
    <Stage>
      <Network />
      <Note>{tr(bi('input → many small machines → output', 'ورودی ← چند ماشین کوچک ← خروجی'))}</Note>
    </Stage>
  );
}

function StepGraph({ step, x, setX }: { step: number; x: number; setX: (x: number) => void }) {
  const tick = useTicker(step === 4, TABLE.length, 700);
  return (
    <Stage>
      <div style={{ transform: 'scale(0.7)', margin: '-16px 0' }}>
        <Machine input={step === 5 ? fmt(x, 2) : 2} output={step === 5 ? fmt(rule(x), 2) : 6} rule="f(x) = 3x" inputLabel="x" outputLabel="f(x)" />
      </div>
      <Graph dots={step === 4 ? tick : TABLE.length} line={step === 5} x={x} onDrag={step === 5 ? setX : undefined} />
    </Stage>
  );
}

export const machineLesson: StoryLesson = {
  id: ID,
  Scene,
  steps: [
    {
      say: bi('Imagine a machine: you give it a number, and it gives you a number back.', 'فرض کن یک ماشین داریم: یک عدد بهش می‌دهی، یک عدد پس می‌دهد.'),
      more: bi(
        'That is all a machine does here: a number goes in on the left, a number comes out on the right. We don\'t yet know what happens inside the box.',
        'کار این ماشین فقط همین است: یک عدد از چپ وارد می‌شود و یک عدد از راست بیرون می‌آید. فعلاً نمی‌دانیم داخل جعبه چه اتفاقی می‌افتد.',
      ),
    },
    {
      say: bi('This machine has one rule: "times three". Put in 2, get out 6.', 'این ماشین یک قاعده دارد: «سه برابرش کن». ۲ می‌دهیم، ۶ می‌گیریم.'),
      more: bi(
        'The rule is what the machine does with every number it receives. Here it multiplies by 3, so 2 becomes 3 × 2 = 6.',
        'قاعده یعنی کاری که ماشین با هر عددی که می‌گیرد انجام می‌دهد. این‌جا ضرب در ۳ است، پس ۲ می‌شود ۳ × ۲ = ۶.',
      ),
    },
    {
      say: bi('Change the input and the output changes. The rule stays the same.', 'ورودی را عوض کن، خروجی هم عوض می‌شود. قاعده همان است.'),
      more: bi(
        '0 gives 0, 1 gives 3, 2 gives 6, 3 gives 9. The same rule works for every number, so we can write all the answers in a table.',
        '۰ می‌شود ۰، ۱ می‌شود ۳، ۲ می‌شود ۶، ۳ می‌شود ۹. یک قاعده برای همه عددها کار می‌کند، پس می‌توانیم جواب‌ها را در یک جدول بنویسیم.',
      ),
    },
    {
      say: bi('We call the input x and write the machine as $f(x) = 3x$: "the machine\'s output when the input is x".', 'به ورودی می‌گوییم x و کار ماشین را این‌طور می‌نویسیم: $f(x) = 3x$ — یعنی «خروجی ماشین، وقتی ورودی x است».'),
      more: bi(
        'x is just a name for "whatever number goes in", like an empty box. f is the name of the machine. $f(2) = 6$ is read "f of 2 is 6": put 2 in, get 6 out. $3x$ means 3 times x.',
        'x فقط یک اسم است برای «هر عددی که وارد می‌شود»، مثل یک جعبه خالی. f اسم ماشین است. $f(2) = 6$ را می‌خوانیم «اِف ِ دو برابر شش»: ۲ را بده، ۶ بگیر. $3x$ یعنی ۳ ضرب در x.',
      ),
    },
    {
      say: bi('Each input–output pair can be a dot: the input goes right, the output goes up.', 'هر جفت ورودی و خروجی را می‌شود یک نقطه کرد: ورودی به راست، خروجی به بالا.'),
      more: bi(
        'For the pair (2, 6): walk 2 steps to the right, then 6 steps up, and put a dot there. Every row of the table becomes one dot.',
        'برای جفت (۲، ۶): ۲ قدم به راست برو، بعد ۶ قدم بالا، و آن‌جا یک نقطه بگذار. هر سطر جدول یک نقطه می‌شود.',
      ),
    },
    {
      say: bi('Put all the dots together and you get a picture of the whole machine: its graph.', 'همه نقطه‌ها را کنار هم بگذار: یک تصویر از کل ماشین داری. به این می‌گوییم نمودار.'),
      more: bi(
        'Between the dots there are more inputs (1.5, 2.7, …) and each has its dot too. Together they form this line. Drag the blue point: the line tells you the output for any input without computing.',
        'بین نقطه‌ها هم ورودی‌های دیگری هست (۱٫۵، ۲٫۷، …) و هر کدام نقطه خودش را دارد. همه با هم این خط را می‌سازند. نقطه آبی را بکش: خط بدون حساب کردن خروجی هر ورودی را نشان می‌دهد.',
      ),
    },
    {
      say: bi('A neural network is the same: a machine that takes a number and gives a number. The difference is that it learns its own rule.', 'شبکه عصبی هم همین است: یک ماشین که عدد می‌گیرد و عدد پس می‌دهد. فرقش این است که قاعده‌اش را خودش یاد می‌گیرد.'),
      more: bi(
        'Inside, a network is made of many small machines connected together. Nobody writes its rule by hand; it adjusts itself until its answers are good. The rest of the course shows exactly how.',
        'داخل شبکه، تعداد زیادی ماشین کوچک به هم وصل شده‌اند. هیچ‌کس قاعده‌اش را دستی نمی‌نویسد؛ خودش را تنظیم می‌کند تا جواب‌هایش خوب شود. بقیه این دوره نشان می‌دهد دقیقاً چطور.',
      ),
    },
  ],
};
