# Storyboard

The course as one story in 12 lessons. Every lesson is a sequence of **steps**. On each step, one short sentence appears under the visual (**FA** is the original, **EN** its translation) and the visual changes to show exactly that sentence (**Scene**). The presenter narrates over it; a reader can follow it alone.

The same things carry through every lesson:

- **The machine.** A neural network is a machine that turns a number into a number.
- **Its knobs.** The weights `w` and the bias `b`. Learning means turning the knobs.
- **The running example.** 2 notebooks; the real price is 10 coins. The one-neuron machine with `w = 3` guesses 6, so it is 4 short, and the loss is 16.
- **The four numbers.** Output `ŷ` is what the machine says. Loss `L` is how bad that is. Derivative `dL/dw` is how fast the loss changes when one knob turns. Gradient is the list of those derivatives for all knobs.
- **Callbacks.** Later lessons say *"this is the X from lesson N"* (the `recall` field in the lesson files).

Scene rules: one big visual per lesson that evolves from step to step, values animate when they change (`useTween`), every number is computed, the colours are the semantic colours (input blue, weight purple, bias amber, output green, target gold, loss red, slope magenta).

---

## 1 · ماشینِ عدد (The number machine)

Idea: a function is a machine with a rule; its picture is a graph.

1. **FA** فرض کن یک ماشین داریم: یک عدد بهش می‌دهی، یک عدد پس می‌دهد.
   **EN** Imagine a machine: you give it a number, and it gives you a number back.
   **Scene:** a large machine box in the centre, an empty input slot on the left, an empty output tray on the right.
2. **FA** این ماشین یک قاعده دارد: «سه برابرش کن». ۲ می‌دهیم، ۶ می‌گیریم.
   **EN** This machine has one rule: "times three". Put in 2, get out 6.
   **Scene:** "×3" appears on the box; a blue 2 slides in, the box pulses, a green 6 slides out.
3. **FA** ورودی را عوض کن، خروجی هم عوض می‌شود. قاعده همان است.
   **EN** Change the input and the output changes. The rule stays the same.
   **Scene:** inputs 0, 1, 2, 3 go through one after another; a small table input → output fills in below.
4. **FA** به ورودی می‌گوییم x. کار ماشین را این‌طور می‌نویسیم: f(x) = 3x — یعنی «خروجی ماشین، وقتی ورودی x است».
   **EN** We call the input x and write the machine as f(x) = 3x: "the machine's output when the input is x".
   **Scene:** the input slot is labelled x, the box shows f(x) = 3x, the table header becomes x | f(x).
5. **FA** هر جفت ورودی و خروجی را می‌شود یک نقطه کرد: ورودی به راست، خروجی به بالا.
   **EN** Each input–output pair can be a dot: the input goes right, the output goes up.
   **Scene:** the machine shrinks to the top; a grid appears; each table row flies to its dot (0,0), (1,3), (2,6), (3,9) with dashed guides to the axes.
6. **FA** همه نقطه‌ها را کنار هم بگذار: یک تصویر از کل ماشین. به این می‌گوییم نمودار.
   **EN** Put all the dots together and you get a picture of the whole machine: its graph.
   **Scene:** the line through the dots is drawn; a draggable point slides along it showing (x, 3x). The learner can drag.
7. **FA** شبکه عصبی هم همین است: یک ماشین که عدد می‌گیرد و عدد پس می‌دهد. فرقش این است که قاعده‌اش را خودش یاد می‌گیرد.
   **EN** A neural network is the same: a machine that takes a number and gives a number. The difference is that it learns its own rule.
   **Scene:** the box morphs into a small network silhouette (circles and lines) with the same input and output.

## 2 · پیچ‌های ماشین: نورون (The machine's knobs: a neuron)

Idea: neuron = `ŷ = w·x + b`. `w` = steepness (slope) of its line, `b` = shift.

1. **FA** ماشین ما قیمت حدس می‌زند: تعداد دفترچه را می‌گیرد و قیمت کل را حدس می‌زند. داخلش یک پیچ هست: w، قیمت هر دفترچه.
   **EN** Our machine guesses prices: it takes a number of notebooks and guesses the total price. Inside it has a knob, w: the price of one notebook.
   **Scene:** the machine from lesson 1 with a purple knob labelled w = 3; input 2 notebooks → 6.
2. **FA** پیچ را بچرخان: w بزرگ‌تر یعنی هر دفترچه گران‌تر، و خط تندتر.
   **EN** Turn the knob: a bigger w means each notebook costs more, and the line gets steeper.
   **Scene:** the knob animates 3 → 5 → 1 → 3; the graph line on the right rotates with it; the output updates.
3. **FA** تندی خط یعنی: یک دفترچه بیشتر، قیمت چقدر بالا می‌رود؟ دقیقاً w. به این «شیب» می‌گوییم — یادت بماند.
   **EN** The steepness of the line answers: one more notebook, how much more money? Exactly w. We call this the slope. Remember it.
   **Scene:** a step triangle on the line: one step right (+1), w up (+3), labelled "slope = 3".
4. **FA** پیچ دوم: b، یک هزینه ثابت مثل پیک. خط را بالا و پایین می‌برد ولی تندی‌اش را عوض نمی‌کند.
   **EN** A second knob: b, a fixed fee like delivery. It moves the line up and down without changing its steepness.
   **Scene:** an amber knob b animates 0 → 2 → 0; the line shifts parallel; the step triangle stays the same.
5. **FA** پس کار ماشین این است: ŷ = w·x + b. کلاه روی y یعنی «حدس».
   **EN** So the machine computes ŷ = w·x + b. The hat on y means "a guess".
   **Scene:** the formula assembles from coloured pieces, then fills in: 3 · 2 + 0 = 6.
6. **FA** این ماشین کوچک را «نورون» می‌نامیم: یک ورودی، یک وزن، یک بایاس، یک خروجی.
   **EN** We call this little machine a neuron: one input, one weight, one bias, one output.
   **Scene:** the box turns into the neuron diagram: input circle → arrow labelled w → circle with "+b" → ŷ.
7. **FA** اگر چند ورودی داشته باشد، هر ورودی وزن خودش را دارد و همه با هم جمع می‌شوند.
   **EN** With several inputs, each input has its own weight, and everything is added up.
   **Scene:** a second input (pens, x₂ = 1, w₂ = 2) joins; bars stack: 3·2 + 2·1 = 8.
8. **FA** یاد گرفتن یعنی پیدا کردن عددهای درست برای این پیچ‌ها. ولی قبلش ببینیم با وصل کردن نورون‌ها چه می‌شود.
   **EN** Learning means finding the right numbers for these knobs. First, let's see what happens when we connect neurons.
   **Scene:** sliders w and b appear for free play; the line and output follow.

## 3 · خم کردن و ساختن شبکه (Bending and building a network)

Idea: ReLU bends the line; neurons in layers; the forward pass.

1. **FA** دو نورون را پشت سر هم بگذار: خروجی اولی، ورودی دومی است. ۲ ← ×۳ ← ۶ ← ×۲ ← ۱۲.
   **EN** Put two neurons in a row: the first one's output is the second one's input. 2 → ×3 → 6 → ×2 → 12.
   **Scene:** two neuron circles in a row; the value 2 travels and becomes 6, then 12.
2. **FA** یک مشکل: دو خط راست پشت سر هم، باز هم یک خط راست است. ×۳ و بعد ×۲ همان ×۶ است.
   **EN** One problem: two straight lines in a row are still a straight line. ×3 then ×2 is just ×6.
   **Scene:** a graph of the chain's output against x: a straight line, labelled "= one neuron ×6".
3. **FA** راه‌حل: بین‌شان یک «خم» بگذار. ReLU: عدد منفی را صفر کن، عدد مثبت را دست نزن.
   **EN** The fix: put a bend between them. ReLU: turn negative numbers into 0, leave positive numbers alone.
   **Scene:** the ReLU graph: flat at 0 on the left, a 45° line on the right; a dot slides across showing input → output.
4. **FA** حالا خروجی نورون یک خط شکسته است: یک‌جا صاف، یک‌جا بالارونده.
   **EN** Now the neuron's output is a bent line: flat in one part, rising in the other.
   **Scene:** one neuron with bias −2: the line w·x + b is drawn faint, ReLU of it bold (a hinge at x = 2/3… computed).
5. **FA** چند تا از این خط‌های شکسته را جمع کن، تقریباً هر شکلی می‌سازی. قدرت شبکه از همین‌جاست.
   **EN** Add a few of these bent lines and you can build almost any shape. That is where a network's power comes from.
   **Scene:** 4 coloured hinges appear one by one and their sum traces a U-shaped curve.
6. **FA** نورون‌ها را ستون‌ستون می‌چینیم؛ هر ستون یک «لایه» است.
   **EN** We arrange neurons in columns; each column is a layer.
   **Scene:** a network diagram: input layer, a hidden layer, an output layer, labelled.
7. **FA** عدد از چپ وارد می‌شود و لایه‌به‌لایه جلو می‌رود تا خروجی. به این «انتشار رو به جلو» می‌گوییم.
   **EN** A number enters on the left and moves forward layer by layer to the output. This is the forward pass.
   **Scene:** the 2-neuron chain network (preset chain-two): the values light up in order: x = 2, 6, 6, 12, ŷ = 12.
8. **FA** شبکه ما برای ۲ دفترچه می‌گوید ۱۲. ولی قیمت واقعی ۱۰ بود. چقدر اشتباه کرده؟
   **EN** For 2 notebooks our network says 12. But the real price was 10. How wrong is it?
   **Scene:** the output 12 next to a gold "10 (real)" tag, with a question mark.

## 4 · چقدر اشتباه کرده؟ (How wrong is it? The loss)

Idea: the loss is one number for "how bad"; it is not the output.

1. **FA** برگردیم سراغ ساده‌ترین ماشین: یک نورون با w = ۳. برای ۲ دفترچه می‌گوید ۶. قیمت واقعی ۱۰ است.
   **EN** Back to the simplest machine: one neuron with w = 3. For 2 notebooks it says 6. The real price is 10.
   **Scene:** a number line: a green dot at 6 (guess) and a gold dot at 10 (real).
2. **FA** به جواب درست می‌گوییم y. اختلاف = حدس منهای جواب: ۶ − ۱۰ = −۴. یعنی ۴ تا کم.
   **EN** We call the right answer y. The error is guess minus answer: 6 − 10 = −4, four too few.
   **Scene:** an arrow from 10 to 6 labelled −4.
3. **FA** ۴ تا کم یا ۴ تا زیاد، هر دو به یک اندازه بدند. علامت مهم نیست، اندازه مهم است.
   **EN** Four too few or four too many: equally bad. The sign doesn't matter, the size does.
   **Scene:** a second ghost guess at 14 (+4) mirrors the first.
4. **FA** اختلاف را به توان دو می‌رسانیم: (−۴)² = ۱۶. همیشه مثبت، و اشتباه بزرگ خیلی بیشتر جریمه می‌شود.
   **EN** We square the error: (−4)² = 16. Always positive, and big mistakes cost much more.
   **Scene:** a square of side 4 grows from the arrow (16 small cells); then side 1 (1 cell) and side 8 (64 cells) for comparison.
5. **FA** به این عدد می‌گوییم خطا (Loss): L = (ŷ − y)². یک عدد که می‌گوید ماشین چقدر بد است. صفر یعنی بی‌نقص.
   **EN** This number is the loss: L = (ŷ − y)². One number that says how bad the machine is. Zero means perfect.
   **Scene:** a red loss meter (gauge) reading 16.
6. **FA** با چند نمونه، خطای هر کدام را حساب می‌کنیم و میانگین می‌گیریم: یک عدد برای کل کار.
   **EN** With several examples, we compute each one's loss and take the average: one number for the whole job.
   **Scene:** 4 receipts (examples) each with its square; the squares stack and level out to the average.
7. **FA** دو عدد را قاطی نکن: خروجی (۶) جوابِ ماشین است؛ خطا (۱۶) نمره بد بودنش.
   **EN** Don't mix up two numbers: the output (6) is the machine's answer; the loss (16) is its badness score.
   **Scene:** two cards side by side: Output ŷ = 6 (green), Loss L = 16 (red).
8. **FA** یادگیری یعنی پیچ‌ها را طوری بچرخان که خطا کم شود. ولی به کدام طرف؟
   **EN** Learning means turning the knobs so the loss goes down. But which way?
   **Scene:** the knob w with a question mark between ← and →; the loss meter beside it.

## 5 · شیب = مشتق (Slope = derivative)

Idea: one concept: how fast the loss changes when you nudge a knob. The slope of lesson 2, now at a point of a curve.

1. **FA** پیچ w را کمی بچرخان و خطا را نگاه کن: w از ۳ به ۳٫۵، خطا از ۱۶ به ۹.
   **EN** Turn the knob w a little and watch the loss: w from 3 to 3.5, loss from 16 to 9.
   **Scene:** the knob and the loss meter side by side; the knob animates 3 → 3.5, the meter 16 → 9.
2. **FA** برای هر w خطا را حساب کن و روی نمودار بگذار: یک منحنی U شکل. ته منحنی بهترین w است.
   **EN** Compute the loss for every w and plot it: a U-shaped curve. The bottom of the curve is the best w.
   **Scene:** the knob sweeps 0 → 8 and the curve L(w) = (2w − 10)² is traced by the moving dot; the bottom (w = 5) is marked.
3. **FA** یادت هست در درس ۲ به تندی خط گفتیم شیب؟ روی خط راست، شیب همه‌جا یکی است.
   **EN** Remember the steepness of the line in lesson 2, the slope? On a straight line the slope is the same everywhere.
   **Scene:** recall card (lesson 2); the straight line with the same step triangle drawn at three places.
   **recall:** lesson 2.
4. **FA** ولی منحنی همه‌جا یک‌جور تند نیست: این‌جا تند، آن‌جا آرام. پس شیب را باید «در یک نقطه» پرسید.
   **EN** But a curve isn't equally steep everywhere: steep here, gentle there. So we ask for the slope at a point.
   **Scene:** back on the U curve; small tangent segments appear at w = 1, 3, 5, 7 with different steepness.
5. **FA** روی w = ۳ زوم کن... باز هم... از نزدیک، منحنی یک خط راست به نظر می‌رسد. شیبِ این خط، شیب منحنی در همان نقطه است.
   **EN** Zoom in at w = 3… and again… Up close the curve looks like a straight line. The slope of that line is the curve's slope at that point.
   **Scene:** a smooth zoom (×1 → ×10 → ×100) centred on w = 3 until the curve is straight; a step triangle appears on it.
6. **FA** حسابش ساده است: یک تلنگر کوچک به w بزن. w از ۳ به ۳٫۰۰۱، خطا از ۱۶ به ۱۵٫۹۸۴. تغییر خطا ÷ تلنگر ≈ −۱۶.
   **EN** It's easy to compute: give w a tiny nudge. w 3 → 3.001, loss 16 → 15.984. Change in loss ÷ nudge ≈ −16.
   **Scene:** at max zoom: Δw = 0.001 and ΔL = −0.016 drawn as the triangle; the division animates to −16.
7. **FA** به این عدد می‌گوییم مشتق، dL/dw: «با یک تلنگر کوچک به w، خطا چند برابر تغییر می‌کند؟» شیب و مشتق یک چیزند.
   **EN** This number is the derivative, dL/dw: "for a tiny nudge to w, how many times as much does the loss change?" Slope and derivative are the same thing.
   **Scene:** zoom back out; the tangent line at w = 3 with the label dL/dw = −16.
8. **FA** علامتش راه را نشان می‌دهد: منفی یعنی w را زیاد کنی خطا کم می‌شود؛ مثبت برعکس؛ صفر یعنی ته دره‌ای.
   **EN** Its sign tells you which way to go: negative means increasing w lowers the loss; positive the opposite; zero means you're at the bottom.
   **Scene:** the point is draggable along the curve; the tangent and a sign badge (−, +, 0) follow live. Auto-demo: w = 3 (−16), w = 7 (+16), w = 5 (0).
9. **FA** پس مشتق قطب‌نمای یادگیری است. حالا فقط باید قدم برداریم.
   **EN** So the derivative is learning's compass. Now we just need to take steps.
   **Scene:** a compass needle on the point at w = 3 pointing right (downhill).

## 6 · پایین رفتن از تپه (Walking downhill: gradient descent)

Idea: step against the slope; the learning rate sets the step size.

1. **FA** فکر کن روی یک تپه‌ای و مه همه‌جا را گرفته. فقط شیب زیر پایت را حس می‌کنی. چطور به پایین برسی؟
   **EN** Imagine standing on a hill in thick fog. You can only feel the slope under your feet. How do you get down?
   **Scene:** the U curve with fog (faded except near the walker); a small walker at w = 3.
2. **FA** خلاف شیب قدم بردار. شیب منفی است؟ برو جلو. مثبت است؟ برو عقب.
   **EN** Step against the slope. Slope negative? Go forward. Positive? Go back.
   **Scene:** at w = 3 a tangent and a green arrow pointing right; a ghost walker at w = 7 with an arrow pointing left.
3. **FA** به زبان ریاضی: w جدید = w − η × مشتق. η (اِتا) اندازه قدم است؛ به آن نرخ یادگیری می‌گوییم.
   **EN** In maths: new w = w − η × derivative. η (eta) is the step size, called the learning rate.
   **Scene:** the update rule with coloured parts; η shown as a small ruler.
4. **FA** با η = ۰٫۰۵: w = ۳ − ۰٫۰۵ × (−۱۶) = ۳٫۸. منهای یک عدد منفی یعنی اضافه کردن. خطا از ۱۶ رسید به ۵٫۷۶!
   **EN** With η = 0.05: w = 3 − 0.05 × (−16) = 3.8. Subtracting a negative number adds. The loss drops from 16 to 5.76!
   **Scene:** the walker hops from 3 to 3.8; the loss meter falls 16 → 5.76.
5. **FA** تکرار کن: هر بار مشتق را دوباره حساب کن و یک قدم بردار. نزدیک ته، شیب کم است، پس قدم‌ها کوچک می‌شوند.
   **EN** Repeat: recompute the derivative each time and take another step. Near the bottom the slope is small, so the steps shrink.
   **Scene:** automatic steps (about 1 per 0.6 s) with a trail; a small table of steps fills in.
6. **FA** رسیدیم: w = ۵. ماشین برای ۲ دفترچه می‌گوید ۱۰، دقیقاً درست. ماشین «یاد گرفت» هر دفترچه ۵ سکه است.
   **EN** We made it: w = 5. The machine says 10 for 2 notebooks, exactly right. It "learned" that a notebook costs 5.
   **Scene:** the walker rests at the bottom; the machine from lesson 2 appears showing 2 → 10.
7. **FA** اگر η خیلی کوچک باشد، قدم‌ها ریزند و رسیدن خیلی طول می‌کشد.
   **EN** If η is too small, the steps are tiny and it takes forever.
   **Scene:** a second run with η = 0.005 crawling.
8. **FA** اگر خیلی بزرگ باشد، از ته دره می‌پری آن طرف... و ممکن است هر بار دورتر شوی.
   **EN** If it's too big, you jump right over the bottom… and may land farther away each time.
   **Scene:** a run with η = 0.27 zig-zagging outward (clipped at the plot edge, labelled "flies off").
9. **FA** خودت امتحان کن. پیدا کردن η خوب، بخش مهمی از آموزش شبکه است.
   **EN** Try it yourself. Finding a good η is an important part of training a network.
   **Scene:** an η slider, Step / Run / Reset buttons.

## 7 · چند پیچ با هم: گرادیان (Many knobs: the gradient)

Idea: one derivative per knob; the list of them is the gradient.

1. **FA** ماشین واقعی چند پیچ دارد. پیچ b را برگردانیم. حالا خطا به دو چیز بستگی دارد: w و b.
   **EN** A real machine has several knobs. Let's bring back knob b. Now the loss depends on two things: w and b.
   **Scene:** two knobs w = 3, b = 0 and the loss meter at 16.
2. **FA** به جای منحنی، یک زمین داریم: هر نقطه‌اش یک w و یک b است و ارتفاعش خطا.
   **EN** Instead of a curve we have a landscape: each point is a w and a b, and its height is the loss.
   **Scene:** a 3-D surface of L(w, b) = (2w + b − 10)², slowly rotating; a dot at (3, 0, 16).
3. **FA** از بالا نگاهش کن، مثل نقشه کوهنوردی: هر خط، جاهای هم‌ارتفاع را به هم وصل می‌کند.
   **EN** Look from above, like a hiking map: each line joins places of equal height.
   **Scene:** the camera tilts to top view and becomes the contour map; the dot stays.
4. **FA** فقط w را تکان بده، b ثابت: شیب در جهت w = −۱۶. همان مشتق درس ۵.
   **EN** Nudge only w, keep b fixed: the slope in the w direction is −16. The same derivative as in lesson 5.
   **Scene:** a horizontal slice through the dot (b fixed) is lifted out as the familiar U curve with the tangent −16.
   **recall:** lesson 5.
5. **FA** حالا فقط b را تکان بده: شیب در جهت b = −۸.
   **EN** Now nudge only b: the slope in the b direction is −8.
   **Scene:** the vertical slice (w fixed) with its tangent −8.
6. **FA** به هر کدام «مشتق جزئی» می‌گوییم (∂ یعنی بقیه ثابت‌اند). فهرست هر دو با هم: گرادیان = (−۱۶، −۸).
   **EN** Each is a partial derivative (∂ means the others are held still). The list of both is the gradient: (−16, −8).
   **Scene:** ∂L/∂w = −16 and ∂L/∂b = −8 combine into ∇L = (−16, −8), drawn as an arrow on the map pointing uphill.
7. **FA** گرادیان کاهشی همان است، برای همه پیچ‌ها با هم: هر پیچ خلاف شیب خودش قدم برمی‌دارد.
   **EN** Gradient descent is the same, for all knobs at once: each knob steps against its own slope.
   **Scene:** steps animate on the map from (3, 0) towards the valley floor; the loss meter falls.
8. **FA** یک شبکه واقعی میلیون‌ها پیچ دارد، پس گرادیانش میلیون‌ها عدد است. ولی این همه مشتق را چطور حساب کنیم؟
   **EN** A real network has millions of knobs, so its gradient has millions of numbers. But how do we compute so many derivatives?
   **Scene:** the map fades into a large network silhouette with many knobs blinking.

## 8 · زنجیره (The chain rule)

Idea: through connected steps, rates multiply.

1. **FA** سه چرخ‌دنده: اولی یک دور بزند، دومی ۳ دور می‌زند. هر دور دومی، سومی را ۲ دور می‌چرخاند. اولی یک دور ← سومی ۳ × ۲ = ۶ دور.
   **EN** Three gears: turn the first once and the second turns 3 times. Each turn of the second turns the third twice. One turn of the first → 3 × 2 = 6 turns of the third.
   **Scene:** three animated gears with their rates 3 and 2; a counter shows 6.
2. **FA** قانون زنجیره: وقتی چیزها پشت سر هم وصل‌اند، نرخ‌ها در هم ضرب می‌شوند.
   **EN** The chain rule: when things are connected in a row, their rates multiply.
   **Scene:** the gear rates written as a product: 3 × 2 = 6.
3. **FA** برگردیم سراغ شبکه دو نورونی درس ۳: ۲ ← ۶ ← ۱۲. جواب درست ۱۰ است، پس خطا = (۱۲ − ۱۰)² = ۴.
   **EN** Back to the two-neuron network from lesson 3: 2 → 6 → 12. The right answer is 10, so the loss is (12 − 10)² = 4.
   **Scene:** the chain-two network with values; the loss node shows 4.
   **recall:** lesson 3.
4. **FA** اگر w۱ را کمی تکان بدهیم، خطا چقدر عوض می‌شود؟ w۱ مستقیم به خطا وصل نیست؛ از یک زنجیره رد می‌شود.
   **EN** If we nudge w₁ a little, how much does the loss change? w₁ isn't wired straight to the loss; it goes through a chain.
   **Scene:** the path w₁ → first neuron → output → loss highlighted, drawn like the gears.
5. **FA** حلقه آخر: اگر خروجی یک تلنگر بخورد، خطا ۴ برابرش تغییر می‌کند. (۲ × (۱۲ − ۱۰) = ۴)
   **EN** Last link: nudge the output and the loss changes 4 times as much. (2 × (12 − 10) = 4)
   **Scene:** the output → loss link shows "×4".
6. **FA** حلقه وسط: نورون دوم ضرب در ۲ می‌کند، پس خروجی ۲ برابرِ تلنگر نورون اول تغییر می‌کند.
   **EN** Middle link: the second neuron multiplies by 2, so the output changes twice as much as the first neuron does.
   **Scene:** the middle link shows "×2" (w₂ = 2, ReLU on: ×1).
7. **FA** حلقه اول: ورودی ۲ است، پس هر تلنگر به w۱ در نورون اول ۲ برابر می‌شود.
   **EN** First link: the input is 2, so every nudge to w₁ is doubled in the first neuron.
   **Scene:** the first link shows "×2" (x = 2).
8. **FA** ضرب کن: ۴ × ۲ × ۲ = ۱۶. تلنگر واقعی هم همین را می‌گوید. این همان مشتق درس ۵ است، فقط از یک زنجیره رد شده.
   **EN** Multiply: 4 × 2 × 2 = 16. A real nudge agrees. It's the same derivative as in lesson 5, passed along a chain.
   **Scene:** the three rates multiply into dL/dw₁ = 16; a nudge check (w₁ + 0.001 → loss + 0.016) confirms.
   **recall:** lesson 5.

## 9 · پس‌انتشار (Backpropagation)

Idea: one backward sweep, sharing pieces, gives every knob's derivative.

1. **FA** مشتق خطا نسبت به w۲ را هم می‌خواهیم. باید همه را از اول حساب کنیم؟
   **EN** We also want the derivative of the loss with respect to w₂. Do we start over?
   **Scene:** both paths (w₁'s and w₂'s) highlighted; they share the last link.
2. **FA** نه! حلقه آخر (۴) بین هر دو مشترک است. یک بار حساب کن، دو بار استفاده کن.
   **EN** No! The last link (×4) is shared by both. Compute it once, use it twice.
   **Scene:** the shared link glows; "×4" is copied to both paths.
3. **FA** پس از آخر شروع می‌کنیم و عقب‌عقب می‌رویم: هر نورون سهمِ تقصیری را که به او رسیده در نرخ خودش ضرب می‌کند و به عقب می‌فرستد.
   **EN** So we start at the end and walk backwards: each neuron multiplies the blame it receives by its own rate and passes it back.
   **Scene:** a red pulse travels from the loss node back through the network.
4. **FA** ۴ به خروجی می‌رسد. سهم w۲: ۴ × ۶ = ۲۴. به نورون اول: ۴ × ۲ = ۸. سهم w۱: ۸ × ۲ = ۱۶.
   **EN** 4 reaches the output. w₂'s share: 4 × 6 = 24. Passed to the first neuron: 4 × 2 = 8. w₁'s share: 8 × 2 = 16.
   **Scene:** the numbers appear on the edges in backward order: 4, 24, 8, 16.
5. **FA** با یک بار رفتن به عقب، مشتق همه پیچ‌ها را داریم. اسم این روش «پس‌انتشار» است.
   **EN** One trip backwards gives us the derivative of every knob. This method is called backpropagation.
   **Scene:** all knobs show their derivative; the gradient list (16, 24) appears.
   **recall:** lesson 7 (gradient).
6. **FA** اگر یک پیچ از دو راه به خروجی برسد؟ سهم هر دو راه را جمع می‌کنیم. ۳ + ۳ = ۶.
   **EN** What if a knob reaches the output along two paths? We add up both paths' shares: 3 + 3 = 6.
   **Scene:** the branching network; two paths from one weight highlighted with their products 3 and 3, summing to 6; nudge check.
7. **FA** حالا کل چرخه را داریم: جلو برو، خطا را حساب کن، به عقب برگرد، پیچ‌ها را یک قدم جابه‌جا کن.
   **EN** Now we have the whole cycle: go forward, compute the loss, go backward, move the knobs one step.
   **Scene:** a four-phase cycle (Forward → Loss → Backward → Update) lights up phase by phase on the chain network.
8. **FA** این چرخه را هزاران بار تکرار کن: این یعنی آموزش شبکه عصبی.
   **EN** Repeat this cycle thousands of times: that is training a neural network.
   **Scene:** Train runs on the chain network; the loss history falls towards 0.

## 10 · یادگیری از داده‌ها (Learning from data: batches and epochs)

1. **FA** تا اینجا فقط یک نمونه داشتیم. در واقعیت هزاران نمونه داریم.
   **EN** So far we had one example. In reality we have thousands.
   **Scene:** many dots (examples) appear on a plot; one neuron's line through them.
2. **FA** مشتق را برای هر نمونه حساب کن و میانگین بگیر: جهتی که برای همه خوب است.
   **EN** Compute the derivative for each example and average them: a direction that is good for all.
   **Scene:** each dot sends a small arrow; the arrows average into one big arrow.
3. **FA** ولی حساب همه نمونه‌ها برای هر قدم کند است. به جایش چند نمونه برمی‌داریم: یک «دسته» (batch).
   **EN** But using every example for each step is slow. Instead we take a few: a batch.
   **Scene:** 2 dots are picked out and highlighted.
4. **FA** جهتِ یک دسته کمی کج است ولی تقریباً درست، مثل چشیدن یک قاشق سوپ به جای کل قابلمه.
   **EN** A batch's direction is a little off but roughly right, like tasting one spoonful instead of the whole pot.
   **Scene:** the batch arrow next to the full-average arrow (ghost), close but not equal.
5. **FA** وقتی همه دسته‌ها یک بار دیده شدند، یک «دوره» (epoch) تمام شده.
   **EN** When every batch has been seen once, one epoch is done.
   **Scene:** chips for 4 batches light up one by one; "epoch 1" completes.
6. **FA** چند دوره که بگذرد، خط روی داده‌ها می‌نشیند.
   **EN** After a few epochs the line settles onto the data.
   **Scene:** training runs; the line moves into place.
7. **FA** نمودار خطا پایین می‌آید، کمی لرزان، چون هر دسته فرق دارد.
   **EN** The loss falls during training, a little bumpy because every batch is different.
   **Scene:** the loss history plot.

## 11 · قدم‌های بهتر، شروع درست (Better steps, a good start)

1. **FA** زمین واقعی همیشه یک کاسه گرد نیست. در یک دره باریک، گرادیان کاهشی زیگزاگ می‌زند.
   **EN** A real landscape is not always a round bowl. In a narrow valley, gradient descent zig-zags.
   **Scene:** a narrow valley contour map; the plain gradient descent path zig-zags.
   **recall:** lesson 7.
2. **FA** به جای قدم زدن، یک توپ رها کن: سرعت می‌گیرد و زیگزاگ‌ها را صاف می‌کند. به این «مومنتوم» می‌گوییم.
   **EN** Instead of walking, let a ball roll: it builds up speed and smooths out the zig-zags. This is momentum.
   **Scene:** the momentum path rolls smoothly down the valley.
3. **FA** هر قدم = ۰٫۹ قدم قبلی + شیب جدید. حرکت قبلی را فراموش نمی‌کند.
   **EN** Each step = 0.9 × the previous step + the new slope. It doesn't forget where it was going.
   **Scene:** step arrows: the previous step (faded), its 0.9 share, plus the new slope, giving the new step.
4. **FA** روش Adam کار دیگری هم می‌کند: به هر پیچ اندازه قدم خودش را می‌دهد. امروز بیشتر شبکه‌ها با Adam آموزش می‌بینند.
   **EN** Adam does one more thing: it gives each knob its own step size. Most networks today train with Adam.
   **Scene:** the Adam path added; all three paths race; a step counter.
5. **FA** یک سؤال دیگر: پیچ‌ها از کجا شروع کنند؟ اگر همه یکسان شروع کنند، همه نورون‌ها یکسان یاد می‌گیرند و تا آخر کپی هم می‌مانند.
   **EN** Another question: where do the knobs start? If they all start the same, all neurons learn the same thing and stay copies forever.
   **Scene:** 4 hidden neurons all starting at 0.5: their curves overlap during training.
6. **FA** پس با عددهای کوچک و تصادفی شروع می‌کنیم تا هر نورون راه خودش را برود.
   **EN** So we start with small random numbers, and each neuron finds its own role.
   **Scene:** the same network with random starts: the 4 curves separate and together fit the data.

## 12 · حفظ کردن یا یاد گرفتن؟ (Memorising or learning?)

1. **FA** داده‌های واقعی کمی نویز دارند، مثل ترازویی که هر بار کمی فرق نشان می‌دهد.
   **EN** Real data is a little noisy, like a scale that reads slightly differently every time.
   **Scene:** a smooth true curve (dashed) and noisy training dots around it.
2. **FA** یک شبکه خیلی بزرگ می‌تواند از همه نقطه‌ها رد شود، حتی از نویزها. خطای آموزش تقریباً صفر!
   **EN** A very big network can pass through every dot, noise included. Training loss almost zero!
   **Scene:** a wiggly network curve through every training dot; the training loss meter near 0.
3. **FA** ولی روی داده‌های جدید بد است: حفظ کرده، یاد نگرفته. به این «بیش‌برازش» (overfitting) می‌گوییم.
   **EN** But on new data it's bad: it memorised instead of learning. This is overfitting.
   **Scene:** new (hollow) validation dots appear far from the wiggles; validation loss meter high.
4. **FA** برای همین بخشی از داده را کنار می‌گذاریم و فقط با آن می‌سنجیم. وقتی خطای آن بالا رفت، آموزش را متوقف کن.
   **EN** That's why we set some data aside and only use it to check. When its loss starts rising, stop training.
   **Scene:** training and validation loss curves over time; the lowest validation point marked "stop here".
5. **FA** وقتی آموزش تمام شد، پیچ‌ها قفل می‌شوند. فقط ورودی می‌دهیم و جواب می‌گیریم: این «استفاده» (inference) است.
   **EN** When training is done, the knobs are locked. We just give inputs and get answers: inference.
   **Scene:** the machine from lesson 1 with locked knobs; new inputs go in, answers come out; no loss, no derivative.
   **recall:** lesson 1.
6. **FA** کل مسیر را مرور کن: ماشین، پیچ، شبکه، خطا، مشتق، پایین رفتن، گرادیان، زنجیره، پس‌انتشار، آموزش. همه شبکه‌های عصبی، کوچک یا بزرگ، با همین چند ایده کار می‌کنند.
   **EN** Look back at the whole road: machine, knobs, network, loss, derivative, walking downhill, gradient, chain, backpropagation, training. Every neural network, small or huge, runs on these few ideas.
   **Scene:** the journey map lights up lesson by lesson with a one-word picture for each.
