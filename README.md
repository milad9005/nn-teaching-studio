# Neural Network Teaching Studio

**English** · [فارسی](README.fa.md)

A visual, step-by-step course that explains how neural networks learn, from zero, in Persian and English. No maths background is needed beyond arithmetic.

Each lesson is a short sequence of steps. On every step one sentence appears under a large picture, and the picture changes to show exactly that sentence. Click **Next** (or press the arrow key) and the story moves on. It works for reading alone, for teaching a class, and for recording a video.

Every number on screen comes from one small numerical engine that runs in the browser, so what you see is real.

![A lesson step: the derivative](docs/screenshot-guided.png)

| Persian (right-to-left) | Free exploration |
| --- | --- |
| ![Persian lesson](docs/screenshot-fa.png) | ![Free exploration playground](docs/screenshot-explore.png) |

- [Install and run (step by step, no programming needed)](#install-and-run-step-by-step)
- [Troubleshooting](#troubleshooting)
- [Modes](#modes) · [Lessons](#lessons) · [For developers](#for-developers)

## Install and run (step by step)

You don't need to know programming. The app runs on your own computer and opens in your web browser. Setting it up takes about 10 minutes the first time; after that it starts in a few seconds.

**What you need**

- A computer with Windows 10/11, macOS or Linux
- An internet connection (only for the first setup)
- A modern browser: Chrome, Edge, Firefox or Safari

### Step 1: Install Node.js (one time only)

The app is built with JavaScript and needs a free program called **Node.js** to run.

1. Go to **https://nodejs.org**.
2. Download the version marked **LTS** (the recommended one).
3. Open the downloaded file and click **Next / Continue** through the installer, keeping the default options.
4. Check that it worked:
   - **Windows**: open the Start menu, type `cmd` and open **Command Prompt**.
   - **macOS**: press `Cmd + Space`, type `Terminal` and press Enter.
   - Type the line below and press Enter:

     ```bash
     node -v
     ```

   You should see a version such as `v22.11.0`. Any version from `v20` up is fine. If you see "not recognized" or "command not found", restart the computer and try again.

### Step 2: Download the app

**Option A: download a ZIP (easiest)**

1. At the top of this page, click the green **Code** button, then **Download ZIP**.
2. Unzip it:
   - **Windows**: right-click the ZIP file → **Extract All…** → **Extract**.
   - **macOS**: double-click the ZIP file.
3. You now have a folder called `nn-teaching-studio-main`. Put it somewhere easy to find, for example on your Desktop.

**Option B: with Git (if you already use it)**

```bash
git clone https://github.com/milad9005/nn-teaching-studio.git
```

### Step 3: Start the app

**Windows**

1. Open the `nn-teaching-studio-main` folder.
2. Double-click **`start-windows.bat`**.
   - If Windows shows "Windows protected your PC", click **More info** → **Run anyway**.
3. A black window opens. The first time, it downloads the app's packages (1–3 minutes). Then your browser opens the app automatically.

**macOS**

1. Open **Terminal** (`Cmd + Space`, type `Terminal`, Enter).
2. Type `bash` followed by a space. Don't press Enter yet.
3. Drag the **`start-mac.command`** file from the folder into the Terminal window, then press Enter.
4. The first time, it downloads the app's packages (1–3 minutes). Then your browser opens the app automatically.

You can also double-click `start-mac.command`. If macOS says it "cannot be opened" or is "from an unidentified developer", open **System Settings → Privacy & Security**, scroll down and click **Open Anyway**, or just use the Terminal method above.

**Any system, by hand (also Linux)**

Open a terminal **inside the app's folder** and run these two commands:

```bash
npm install
```

```bash
npm run dev
```

Then open **http://localhost:5178** in your browser.

> To open a terminal inside the folder: on Windows, click the folder's address bar, type `cmd` and press Enter. On macOS, type `cd ` (with a space) in Terminal, drag the folder into the window and press Enter.

### Step 4: Stop and start again

- **To stop** the app, close the terminal window (or press `Ctrl + C` in it). Closing only the browser tab does not stop it.
- **To start it again later**, repeat Step 3. The package download only happens the first time, so it starts in a few seconds.
- The app works offline once installed. Your language, theme, lesson progress and panel layout are saved in the browser.

## Troubleshooting

| What you see | What to do |
| --- | --- |
| `'node' is not recognized…` or `command not found: npm` | Node.js is not installed, or the terminal was opened before installing it. Install Node.js (Step 1), then close and reopen the terminal. If it still fails, restart the computer. |
| PowerShell says `running scripts is disabled on this system` | Use **Command Prompt** (`cmd`) instead of PowerShell, or double-click `start-windows.bat`. |
| `Error: Port 5178 is already in use` | The app is already running in another terminal window. Just open http://localhost:5178, or close the other window and try again. |
| The browser does not open by itself | Open **http://localhost:5178** yourself. |
| `npm install` fails with network errors | Check your internet connection and try again. On a work network a proxy or firewall may block it. |
| `npm install` fails in some other way | Delete the `node_modules` folder inside the app's folder and run `npm install` again. Never use `sudo` with it. |
| Node.js version is too old (`v18` or lower) | Install the current LTS from https://nodejs.org; it replaces the old one. |
| macOS: `permission denied` when double-clicking `start-mac.command` | Use the Terminal method (`bash` + drag the file). |
| The page is blank or looks broken | Refresh with `Ctrl + Shift + R` (Windows) or `Cmd + Shift + R` (macOS) and use an up-to-date browser. |
| You want to reset the layout, language and progress | In the browser, clear the site data for `localhost:5178`, then refresh. |

## Using the app

- **Lessons**: start at lesson 1. **Next** / **Previous** (or the arrow keys and Space) move through the steps; at the end of a lesson, **Next** goes on to the next one. The bar at the top shows all 12 lessons and where you are.
- **Explanation** (the panel beside the picture): every step's sentence plus a fuller explanation, for reading on your own. Close it if you don't need it.
- **Language**: the `فارسی` / `English` button switches everything, including the right-to-left layout. In Persian, the left arrow is "next".
- **Free exploration**: a playground to build your own network, pick a ready-made example and train it.
- **Present**: hides the menus for teaching or recording.

### Presentation keys

| Key | Action |
| --- | --- |
| `→` / `Space` (Persian: `←` / `Space`) | Next step |
| `←` (Persian: `→`) | Previous step |
| `PgDn` / `PgUp` | Next / previous lesson |
| `T` | Hide or show the sentence under the picture (narrate over a clean picture) |
| `L` | Laser pointer |
| `D` | Draw on the screen, `C` clears it |
| `F` | Full screen |
| `Esc` | Leave presentation |

## Lessons

One story in 12 lessons. The same machine, the same knobs and the same example (2 notebooks, a real price of 10 coins) carry through all of them, and later lessons point back to the earlier ones they build on.

1. **The number machine**: a function is a machine with a rule; its picture is a graph.
2. **The machine's knobs: a neuron**: `ŷ = w·x + b`; the weight is the slope of the line, the bias shifts it.
3. **Bending and building a network**: ReLU, layers, the forward pass.
4. **How wrong is it? The loss**: error, squaring, averaging; the loss is not the output.
5. **Slope = derivative**: nudge a knob and watch the loss; zoom into the curve until it is a straight line.
6. **Walking downhill: gradient descent**: step against the slope; the learning rate.
7. **Many knobs: the gradient**: one slope per knob; the loss landscape.
8. **The chain rule**: gears; rates multiply through a network.
9. **Backpropagation**: one trip backwards gives every knob's slope.
10. **Learning from data**: batches and epochs.
11. **Better steps, a good start**: momentum, Adam, random starting weights.
12. **Memorising or learning?**: overfitting, validation, using the trained network.

The full script, every step in Persian and English with what the picture shows, is in [docs/storyboard.md](docs/storyboard.md).

## For developers

Requirements: Node 20+ and npm. The end-to-end tests use the locally installed Google Chrome, so no browser download is needed.

| Command | What it does |
| --- | --- |
| `npm run dev` | Development server at http://localhost:5178 |
| `npm test` | Unit tests (Vitest): engine math, gradients vs finite differences, presets, optimizers, lesson numbers, course structure |
| `npm run e2e` | End-to-end tests (Playwright, local Chrome): every step of every lesson in both languages, keys, presentation, playground |
| `npm run typecheck` | TypeScript check |
| `npm run build` | Typecheck and production build into `dist/` |

### Architecture

```
src/
  engine/        pure TypeScript math, no React. Unit-tested.
    activations  linear, ReLU, sigmoid, tanh, softmax (+ derivatives / Jacobian)
    losses       MSE, MAE, Huber, BCE, CCE (+ gradients)
    network      dense nets: forward, backward, batch loss/grad, params, architecture edits, validation
    analysis     finite-difference gradient check, chain-rule path decomposition
    optimizers   SGD, Momentum, Adam on flat vectors (+ masks, clipping, NaN guards)
    init         zeros, constant, uniform, Xavier, He (seeded)
    training     batching schedule, epochs, one training iteration, divergence guard
    presets      the 12 named teaching scenarios
  state/
    lab.ts       the shared network workspace (Zustand). All network panels read it.
    lessonState  lesson-local state (survives language switches)
    ui.ts        language, theme, mode, lesson, step, panels, presentation (persisted)
  i18n/          interface strings (en, fa) and preset text; lesson prose lives with lessons
  components/    Tex (KaTeX), Plot2D (SVG), PlotlyChart (3-D), NetworkCanvas (React Flow),
                 NumberLine, Inspector, lab panels, controls, prose (inline math in text)
  story/         the course: curriculum.ts (order and titles), StoryView (picture + sentence +
                 steps), kit.tsx (shared visuals: machine, knob, meter, cards), useTween
                 (animation), lessons/NN-id.tsx (one file per lesson) with math-id.ts helpers
  shell/         top bar, playground workspace, presentation layer
e2e/             Playwright tests
```

### One source of truth

- Network lessons and the playground read `useLab` (state) and `useLabDerived()` (forward trace, loss and gradients, memoized from the state). Nothing stores a copy of a computed value.
- A lesson scene draws the state for its current `step` and computes every number it shows with the engine or its `math-<id>.ts` helper (unit-tested).

### Conventions

- **Loss normalization**: per-sample loss averages over output units (CCE sums over classes); the batch loss is the mean over samples, and the backward pass uses the same 1/N.
- **Squared error**: `L = (ŷ − y)²` without a ½, so `dL/dŷ = 2(ŷ − y)`.
- **ReLU at 0**: not differentiable; the engine uses `ReLU'(0) = 0` (like PyTorch). Gradient checks flag parameters whose finite difference crosses a ReLU kink.
- **Momentum**: `v ← βv + g; θ ← θ − lr·v`. **Adam**: standard with bias correction.
- **Numbers**: values below 1e-12 are shown as 0 (floating-point noise). Numbers and equations stay left-to-right in Persian mode.
- **Limits**: up to 4 inputs, 4 hidden layers, 10 units per layer, 32 examples. Training stops with a message on NaN/Inf or when the loss exceeds 1e8.

### Presets

| Preset | Setup | Verified result |
| --- | --- | --- |
| Single Affine Neuron | 1→1 linear, w=2, b=1, x=1.5, y=5 | z = 4, L = 1 |
| ReLU Activation Threshold | 1→1 ReLU, w=1, b=−1 | outputs 0, 0, 0, 1 for x = −1, 0.5, 1, 2 |
| One vs Multiple Neurons | 1→4 ReLU→1, target 0.5x² | ŷ(2) = 1.5 |
| Temperature Prediction | outside temp → cooling kW, 4 examples | MSE = 0.045 |
| Single-Weight Loss Landscape | L(w) = (2w − 10)² | dL/dw(3) = −16 |
| Two-Weight Loss Landscape | 2→1 linear, bias fixed | L = 37/6, ∇L = (−10/3, −11/3) |
| Chain Rule with Two Sequential Neurons | x=2, w1=3, w2=2, ReLU, y=10 | ŷ=12, L=4, dL/dw1=16, dL/dw2=24, dL/db1=8, dL/db2=4 |
| Backpropagation Through a Branching Graph | 1→2→2→1 | W1[0,0] has 2 paths of 3 each, gradient 6 |
| Gradient Descent Step by Step | lr = 0.05 | w 3 → 3.8, L 16 → 5.76 |
| Learning Rate Comparison | same start | slow / fast / oscillating / diverging (lr > 0.25) |
| Mini-Batch Training | 8 examples, batch 2 | 4 updates per epoch |
| Overfitting Demonstration | 8 noisy points, 14 validation | wide net: train loss ≪ validation loss |

All of these are asserted in `src/engine/training.test.ts` and `src/engine/engine.test.ts`.

### Adding a teaching example

**A new preset**

1. Add an id to `PresetId` and a factory to `PRESETS` in `src/engine/presets.ts` (network, data, loss, optimizer, batch, frozen parameters).
2. Add its name and intended outcome in English and Persian to `src/i18n/presets.ts`.
3. Add a test in `src/engine/training.test.ts` that asserts its expected numbers.

It then appears in the playground's preset picker, and a lesson can load it with `presetId`.

**A new lesson or step**

1. Write the step in [docs/storyboard.md](docs/storyboard.md) first: the Persian sentence, the English sentence and what the picture shows.
2. A lesson is `src/story/lessons/NN-id.tsx` exporting a `StoryLesson` (`src/story/types.ts`): `steps` (each with `say`, a fuller `more` explanation and an optional `recall` of an earlier lesson) and a `Scene` component that draws the picture for a given `step`. Register the id and title in `src/story/curriculum.ts`.
3. Use the shared visuals in `src/story/kit.tsx` (machine, knob, loss meter, value cards, the same colours) so the course looks like one story, and `useTween` so changes animate.
4. Run `npm test` and `npm run e2e`. `src/story/story.test.ts` checks that every step is written in both languages, that recalls only point backwards and that no untaught notation appears.
