import type { PresetId } from '../engine';
import type { Bi } from './index';

export const PRESET_TEXT: Record<PresetId, { name: Bi; outcome: Bi }> = {
  'single-affine': {
    name: { en: 'Single Affine Neuron', fa: 'یک نورون Affine' },
    outcome: {
      en: 'z = w·x + b: the weight scales the input, the bias shifts the result.',
      fa: 'z = w·x + b: وزن ورودی را بزرگ و کوچک می‌کند، بایاس نتیجه را جابه‌جا می‌کند.',
    },
  },
  'relu-threshold': {
    name: { en: 'ReLU Activation Threshold', fa: 'آستانه فعال‌سازی ReLU' },
    outcome: {
      en: 'With w = 1, b = −1 the neuron stays silent until x passes 1.',
      fa: 'با w = 1 و b = −1 نورون تا وقتی x از ۱ رد نشود خاموش می‌ماند.',
    },
  },
  'one-vs-many': {
    name: { en: 'One vs Multiple Neurons', fa: 'یک نورون در برابر چند نورون' },
    outcome: {
      en: 'Four ReLU pieces bend a line into a rough parabola; one ReLU cannot.',
      fa: 'چهار تکه ReLU یک خط را به شکل تقریبی سهمی خم می‌کنند؛ یک ReLU نمی‌تواند.',
    },
  },
  temperature: {
    name: { en: 'Temperature Prediction', fa: 'پیش‌بینی دما و کولر' },
    outcome: {
      en: 'Predict cooling power from outside temperature; see each error before averaging.',
      fa: 'توان سرمایش را از دمای بیرون پیش‌بینی کن؛ خطای هر نمونه را قبل از میانگین ببین.',
    },
  },
  'single-weight': {
    name: { en: 'Single-Weight Loss Landscape', fa: 'منظره خطا با یک وزن' },
    outcome: {
      en: 'L(w) = (2w − 10)²: a parabola with its lowest point at w = 5.',
      fa: 'L(w) = (2w − 10)²: یک سهمی که پایین‌ترین نقطه‌اش w = 5 است.',
    },
  },
  'two-weight': {
    name: { en: 'Two-Weight Loss Landscape', fa: 'منظره خطا با دو وزن' },
    outcome: {
      en: 'A bowl over (w1, w2) with its minimum at (2, 1); the gradient points uphill.',
      fa: 'یک کاسه روی (w1, w2) با کمینه در (2, 1)؛ گرادیان به سمت سربالایی اشاره می‌کند.',
    },
  },
  'chain-two': {
    name: { en: 'Chain Rule with Two Sequential Neurons', fa: 'قاعده زنجیره‌ای با دو نورون پشت سر هم' },
    outcome: {
      en: 'x = 2, w1 = 3, w2 = 2, y = 10 → output 12, L = 4, dL/dw1 = 16, dL/dw2 = 24.',
      fa: 'x = 2، w1 = 3، w2 = 2، y = 10 ← خروجی 12، L = 4، dL/dw1 = 16، dL/dw2 = 24.',
    },
  },
  branching: {
    name: { en: 'Backpropagation Through a Branching Graph', fa: 'Backprop در گراف شاخه‌دار' },
    outcome: {
      en: 'W1[0,0] reaches the output along two paths worth 3 each; its gradient is 3 + 3 = 6.',
      fa: 'W1[0,0] از دو مسیر به خروجی می‌رسد که هر کدام 3 هستند؛ گرادیانش 3 + 3 = 6 است.',
    },
  },
  'gd-step': {
    name: { en: 'Gradient Descent Step by Step', fa: 'نزول گرادیان گام‌به‌گام' },
    outcome: {
      en: 'w = 3, dL/dw = −16, lr = 0.05 → w_new = 3.8 and the loss drops from 16 to 5.76.',
      fa: 'w = 3، dL/dw = −16، lr = 0.05 ← w جدید = 3.8 و خطا از 16 به 5.76 می‌رسد.',
    },
  },
  'lr-compare': {
    name: { en: 'Learning Rate Comparison', fa: 'مقایسه نرخ یادگیری' },
    outcome: {
      en: 'Same start, four learning rates: slow, fast, oscillating, diverging (above 0.25).',
      fa: 'نقطه شروع یکسان، چهار نرخ یادگیری: کند، سریع، نوسانی، واگرا (بالای 0.25).',
    },
  },
  'mini-batch': {
    name: { en: 'Mini-Batch Training', fa: 'آموزش Mini-batch' },
    outcome: {
      en: '8 examples, batches of 2: one epoch is 4 updates, each from a different pair.',
      fa: '۸ نمونه، Batchهای ۲تایی: هر Epoch یعنی ۴ به‌روزرسانی، هر کدام با یک جفت متفاوت.',
    },
  },
  overfit: {
    name: { en: 'Overfitting Demonstration', fa: 'نمایش Overfitting' },
    outcome: {
      en: 'Training loss keeps falling while validation loss stops improving and rises.',
      fa: 'خطای آموزش مدام کم می‌شود اما خطای اعتبارسنجی دیگر بهتر نمی‌شود و بالا می‌رود.',
    },
  },
};
