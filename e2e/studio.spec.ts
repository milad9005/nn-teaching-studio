import { test, expect, type Page } from '@playwright/test';

/** Open the app at a given lesson (0-based) and mode, with a clean UI state. */
async function open(page: Page, lessonIndex = 0, mode: 'guided' | 'explore' = 'guided', lang: 'en' | 'fa' = 'en') {
  await page.addInitScript(
    ([i, m, l]) => {
      if (!sessionStorage.getItem('seeded')) {
        localStorage.setItem('nnts-ui', JSON.stringify({ state: { lessonIndex: i, step: 0, mode: m, lang: l, theme: 'light', explainOpen: true, captions: true }, version: 0 }));
        sessionStorage.setItem('seeded', '1');
      }
    },
    [lessonIndex, mode, lang] as const,
  );
  await page.goto('/');
  await expect(page.getByTestId(mode === 'guided' ? 'scene' : 'panel-center')).toBeVisible();
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

test('presentation mode: big visual, step keys, captions toggle, exit', async ({ page }) => {
  await open(page, 0);
  await page.getByTestId('present').click();
  await expect(page.getByTestId('present-bar')).toBeVisible();
  await expect(page.getByTestId('explain-panel')).toHaveCount(0);
  await expect(page.getByTestId('step-count')).toHaveText('1 / 7');
  await page.keyboard.press('ArrowRight');
  await expect(page.getByTestId('step-count')).toHaveText('2 / 7');
  await page.keyboard.press('t');
  await expect(page.getByTestId('caption')).toHaveCount(0);
  await page.keyboard.press('t');
  await expect(page.getByTestId('caption')).toHaveCount(1);
  await page.keyboard.press('Escape');
  await expect(page.getByTestId('present-bar')).toHaveCount(0);
  await expect(page.getByTestId('explain-panel')).toBeVisible();
});

test('steps move with buttons and keys, and cross into the next lesson', async ({ page }) => {
  await open(page, 0);
  await expect(page.getByTestId('lesson-title')).toHaveText('The Number Machine');
  await page.getByTestId('step-next').click();
  await expect(page.getByTestId('step-count')).toHaveText('2 / 7');
  await page.keyboard.press('ArrowLeft');
  await expect(page.getByTestId('step-count')).toHaveText('1 / 7');
  for (let i = 0; i < 7; i++) await page.getByTestId('step-next').click();
  await expect(page.getByTestId('lesson-title')).toHaveText("The Machine's Knobs: a Neuron");
  await expect(page.getByTestId('step-count')).toHaveText(/^1 \//);
  // Persian reads right to left: the left arrow is "next".
  await page.getByTestId('lang-toggle').click();
  await expect(page.locator('html')).toHaveAttribute('dir', 'rtl');
  await page.keyboard.press('ArrowLeft');
  await expect(page.getByTestId('step-count')).toHaveText(/^2 \//);
  await expect(page.getByTestId('lesson-title')).toHaveText('پیچ‌های ماشین: نورون');
});

test('the explanation panel follows the steps and can be closed', async ({ page }) => {
  await open(page, 0);
  await expect(page.locator('.transcript li')).toHaveCount(1);
  await page.getByTestId('step-next').click();
  await page.getByTestId('step-next').click();
  await expect(page.locator('.transcript li')).toHaveCount(3);
  await page.getByTestId('explain-toggle').click();
  await expect(page.getByTestId('explain-panel')).toHaveCount(0);
});

test('every step of every lesson renders in both languages without errors', async ({ page }) => {
  test.setTimeout(240_000);
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  await open(page, 0);
  const total = 12;
  for (let i = 0; i < total; i++) {
    const [, n] = (await page.getByTestId('step-count').innerText()).split('/').map((v) => Number(v.trim()));
    for (let k = 0; k < n; k++) {
      await expect(page.getByTestId('scene')).toHaveAttribute('data-step', String(k));
      await expect(page.getByTestId('caption')).not.toBeEmpty();
      expect(await page.getByTestId('story').innerText(), `NaN in lesson ${i + 1} step ${k + 1}`).not.toMatch(/\bNaN\b|undefined/);
      if (i < total - 1 || k < n - 1) await page.getByTestId('step-next').click();
    }
    if (i === 5) await page.getByTestId('lang-toggle').click();
  }
  expect(errors).toEqual([]);
});
