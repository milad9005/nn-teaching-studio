import { useEffect } from 'react';
import { useUi } from './state/ui';
import { TopBar } from './shell/TopBar';
import { StoryView } from './story/StoryView';
import { ExploreMode } from './shell/Explore';
import { InkLayer, PresentationBar, usePresentationKeys } from './shell/Presentation';
import { useTrainingLoop } from './components/LabPanels';

function useDocumentAttributes() {
  const lang = useUi((s) => s.lang);
  const theme = useUi((s) => s.theme);
  const presentation = useUi((s) => s.presentation);
  const reduced = useUi((s) => s.reducedMotion);
  useEffect(() => {
    const root = document.documentElement;
    root.lang = lang;
    root.dir = lang === 'fa' ? 'rtl' : 'ltr';
    root.dataset.present = String(presentation);
    root.dataset.motion = reduced ? 'reduced' : 'full';
  }, [lang, presentation, reduced]);
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const apply = () => (document.documentElement.dataset.theme = theme === 'system' ? (mq.matches ? 'dark' : 'light') : theme);
    apply();
    mq.addEventListener('change', apply);
    return () => mq.removeEventListener('change', apply);
  }, [theme]);
}

export default function App() {
  useDocumentAttributes();
  usePresentationKeys();
  useTrainingLoop();
  const mode = useUi((s) => s.mode);
  const presentation = useUi((s) => s.presentation);
  return (
    <div className={`app ${presentation ? 'presenting' : ''}`}>
      {!presentation && <TopBar />}
      <main className="app-main">{mode === 'guided' ? <StoryView /> : <ExploreMode />}</main>
      {presentation && <PresentationBar />}
      {presentation && <InkLayer />}
    </div>
  );
}
