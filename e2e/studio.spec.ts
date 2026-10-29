import { test, expect, type Page } from '@playwright/test';

/** Open the app at a given lesson (0-based) and mode, with a clean UI state. */
async function open(page: Page, lessonIndex = 0, mode: 'guided' | 'explore' = 'guided', lang: 'en' | 'fa' = 'en') {
  await page.addInitScript(
    ([i, m, l]) => {
      if (!sessionStorage.getItem('seeded')) {
        localStorage.setItem('nnts-ui', JSON.stringify({ state: { lessonIndex: i, mode: m, lang: l, theme: 'light' }, version: 0 }));
        sessionStorage.setItem('seeded', '1');
      }
    },
    [lessonIndex, mode, lang] as const,
  );
  await page.goto('/');
  await expect(page.getByTestId('panel-center')).toBeVisible();
}

/** Read the shared lab store through Vite's module graph (same instance as the app). */
const LAB = '/src/state/lab.ts';
const lab = (page: Page) =>
  page.evaluate(async (path) => {
    const m = await import(path);
    const s = m.useLab.getState();
    return { net: s.net, data: s.data, history: s.history, opt: s.opt, presetId: s.presetId } as {
      net: { layers: { W: number[][]; b: number[] }[] };
      data: unknown;
      history: { loss: number }[];
      opt: { lr: number };
      presetId: string;
    };
  }, LAB);

const loadPreset = (page: Page, id: string) =>
  page.evaluate(async ([path, pid]) => (await import(path)).useLab.getState().loadPreset(pid), [LAB, id] as const);

test('core example: chain rule lesson shows the specified forward values and gradients', async ({ page }) => {
  await open(page, 9);
  await expect(page.getByTestId('lesson-head')).toContainText('Chain Rule');
  await expect(page.getByTestId('chain-total')).toHaveText('16');
  await expect(page.getByTestId('loss-node')).toContainText('L 4');
  await expect(page.getByTestId('unit-1-0')).toContainText('a 6');
  await expect(page.getByTestId('unit-2-0')).toContainText('ŷ 12');
  await page.getByRole('button', { name: 'dL/dw₂' }).click();
  await expect(page.getByTestId('chain-total')).toHaveText('24');
});

test('moving w1 into the ReLU zero region zeroes the gradient everywhere', async ({ page }) => {
  await open(page, 9);
  const w1 = page.getByTestId('chain-w1').locator('input[type=number]');
  await w1.fill('-1');
  await w1.press('Enter');
  await expect(page.getByTestId('chain-total')).toHaveText('0');
  await expect(page.getByTestId('loss-node')).toContainText('L 100');
  await expect(page.getByTestId('unit-1-0')).toHaveClass(/dead/);
});

test('language switch keeps the simulation state and flips to RTL', async ({ page }) => {
  await open(page, 9);
  const w1 = page.getByTestId('chain-w1').locator('input[type=number]');
  // w1 = 2.25: a1 = 4.5, y_hat = 9, dL/dy_hat = -2, dL/dw1 = -2 * 2 * 2 = -8
  await w1.fill('2.25');
  await w1.press('Enter');
  await expect(page.getByTestId('chain-total')).toHaveText('−8');
  const before = await page.getByTestId('chain-total').innerText();
  await page.getByTestId('lang-toggle').click();
  await expect(page.locator('html')).toHaveAttribute('dir', 'rtl');
  await expect(page.getByTestId('lesson-head')).toContainText('قاعده زنجیره‌ای');
  await expect(page.getByTestId('chain-total')).toHaveText(before);
  expect((await lab(page)).net.layers[0].W[0][0]).toBe(2.25);
  // Equations stay left-to-right inside RTL text.
  const dir = await page.locator('.katex').first().evaluate((el) => getComputedStyle(el).direction);
  expect(dir).toBe('ltr');
});

test('optimizer step uses the selected learning rate (gradient descent lesson preset)', async ({ page }) => {
  await open(page, 0, 'explore');
  await loadPreset(page, 'gd-step');
  await page.getByTestId('btn-optimizer').click();
  const s = await lab(page);
  expect(s.net.layers[0].W[0][0]).toBeCloseTo(3.8, 10);
  expect(s.history.at(-1)!.loss).toBeCloseTo(5.76, 10);
  // Change lr to 0.1 and step again: w = 3.8 - 0.1 * 4 * (7.6 - 10) = 4.76
  const lr = page.getByTestId('lr-slider').locator('input[type=number]');
  await lr.fill('0.1');
  await lr.press('Enter');
  await page.getByTestId('btn-optimizer').click();
  expect((await lab(page)).net.layers[0].W[0][0]).toBeCloseTo(4.76, 10);
});

test('architecture edits update the graph and parameters; gradient check passes', async ({ page }) => {
  await open(page, 0, 'explore');
  await loadPreset(page, 'branching');
  await expect(page.locator('.react-flow__node')).toHaveCount(1 + 2 + 2 + 1 + 1);
  await page.getByTestId('add-unit-0').click();
  await expect(page.locator('.react-flow__node')).toHaveCount(1 + 3 + 2 + 1 + 1);
  await page.getByTestId('add-layer').click();
  const s = await lab(page);
  expect(s.net.layers.length).toBe(4);
  await page.getByRole('button', { name: 'Gradient check' }).click();
  await page.getByTestId('btn-gradcheck').click();
  await expect(page.getByTestId('gradcheck')).toContainText('All gradients agree');
});

test('forward and backward stepping reveal real values layer by layer', async ({ page }) => {
  await open(page, 0, 'explore');
  await loadPreset(page, 'chain-two');
  await page.getByTestId('btn-forward').click();
  await expect(page.getByTestId('unit-1-0')).toContainText('a ?');
  await page.getByTestId('btn-forward').click();
  await expect(page.getByTestId('unit-1-0')).toContainText('a 6');
  await expect(page.getByTestId('loss-node')).toContainText('L ?');
  await page.getByTestId('btn-forward').click();
  await page.getByTestId('btn-forward').click();
  await expect(page.getByTestId('loss-node')).toContainText('L 4');
  await page.getByTestId('btn-backward').click();
  await expect(page.getByTestId('loss-node')).toContainText('∂L/∂ŷ 4');
  await page.getByTestId('btn-backward').click();
  await expect(page.getByTestId('unit-2-0')).toContainText('∂L/∂z 4');
  await expect(page.getByTestId('unit-1-0')).not.toContainText('∂L/∂z');
  await page.getByTestId('btn-backward').click();
  await expect(page.getByTestId('unit-1-0')).toContainText('∂L/∂z 8');
});

test('derivative lesson: the secant slope converges to the tangent', async ({ page }) => {
  await open(page, 6);
  await page.getByRole('button', { name: 'Loss curve' }).click();
  const a = page.getByTestId('slider-a').locator('input[type=number]');
  await a.fill('5');
  await a.press('Enter');
  await expect(page.getByTestId('exact-deriv')).toHaveText('2');
  await expect(page.getByTestId('secant')).toHaveText('2.5');
  const h = page.getByTestId('slider-h').locator('input[type=number]');
  await h.fill('3');
  await h.press('Enter');
  await expect(page.getByTestId('secant')).toHaveText('2.0005');
});

test('presentation mode: reveal steps, keyboard, exit', async ({ page }) => {
  await open(page, 9);
  await page.getByTestId('present').click();
  await expect(page.getByTestId('present-bar')).toBeVisible();
  await expect(page.locator('.reveal-pending').first()).toBeVisible();
  const pending = await page.locator('.reveal-pending').count();
  await page.keyboard.press('ArrowRight');
  await expect(page.locator('.reveal-pending')).toHaveCount(pending - 1);
  await page.keyboard.press('Escape');
  await expect(page.getByTestId('present-bar')).toHaveCount(0);
  await expect(page.locator('.reveal-pending')).toHaveCount(0);
});

test('panels collapse, reopen and maximize', async ({ page }) => {
  await open(page, 9);
  await page.getByTestId('collapse-left').click();
  await expect(page.getByTestId('panel-left')).toHaveCount(0);
  await page.getByTestId('reopen-left').click();
  await expect(page.getByTestId('panel-left')).toBeVisible();
  await page.getByTestId('max-right').click();
  await expect(page.getByTestId('panel-center')).toHaveCount(0);
  await page.getByTestId('max-right').click();
  await expect(page.getByTestId('panel-center')).toBeVisible();
});

test('every lesson renders in both languages without errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  await open(page, 0);
  const total = 18;
  for (let i = 0; i < total; i++) {
    await expect(page.getByTestId('lesson-head')).toContainText(`${String(i + 1).padStart(2, '0')} / ${total}`);
    await expect(page.getByTestId('panel-center')).toBeVisible();
    if (i === 8 || i === 16) await page.getByTestId('lang-toggle').click();
    if (i < total - 1) await page.getByTestId('lesson-next').click();
  }
  expect(errors).toEqual([]);
});
