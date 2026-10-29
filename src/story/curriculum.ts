// The course: 12 lessons that tell one story. The script is docs/storyboard.md.
import { bi, type Bi } from '../i18n';

export interface LessonMeta {
  id: string;
  title: Bi;
  /** One or two words for the journey bar. */
  short: Bi;
}

export const CURRICULUM: LessonMeta[] = [
  { id: 'machine', title: bi('The Number Machine', 'ماشینِ عدد'), short: bi('Machine', 'ماشین') },
  { id: 'neuron', title: bi("The Machine's Knobs: a Neuron", 'پیچ‌های ماشین: نورون'), short: bi('Neuron', 'نورون') },
  { id: 'network', title: bi('Bending and Building a Network', 'خم کردن و ساختن شبکه'), short: bi('Network', 'شبکه') },
  { id: 'loss', title: bi('How Wrong Is It? The Loss', 'چقدر اشتباه کرده؟ خطا'), short: bi('Loss', 'خطا') },
  { id: 'derivative', title: bi('Slope = Derivative', 'شیب = مشتق'), short: bi('Derivative', 'مشتق') },
  { id: 'descent', title: bi('Walking Downhill: Gradient Descent', 'پایین رفتن از تپه: گرادیان کاهشی'), short: bi('Downhill', 'پایین رفتن') },
  { id: 'gradient', title: bi('Many Knobs: the Gradient', 'چند پیچ با هم: گرادیان'), short: bi('Gradient', 'گرادیان') },
  { id: 'chain', title: bi('The Chain Rule', 'قاعده زنجیره‌ای'), short: bi('Chain', 'زنجیره') },
  { id: 'backprop', title: bi('Backpropagation', 'پس‌انتشار'), short: bi('Backprop', 'پس‌انتشار') },
  { id: 'data', title: bi('Learning from Data: Batches and Epochs', 'یادگیری از داده‌ها: دسته و دوره'), short: bi('Data', 'داده') },
  { id: 'better', title: bi('Better Steps, a Good Start', 'قدم‌های بهتر، شروع درست'), short: bi('Better steps', 'قدم بهتر') },
  { id: 'generalize', title: bi('Memorising or Learning?', 'حفظ کردن یا یاد گرفتن؟'), short: bi('Real use', 'استفاده') },
];

export const lessonIndexById = new Map(CURRICULUM.map((l, i) => [l.id, i]));
