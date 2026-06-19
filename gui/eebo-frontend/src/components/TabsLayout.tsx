// src/components/TabsLayout.tsx

import { children, createSignal, Show } from "solid-js";


export type Tab = {
  key: string;
  label: string;
  icon: string;
  component: any;
};

export function Tabs(props: { tabs: Tab[] }) {
  const [active, setActive] = createSignal(props.tabs[0].key);

  return (
    <div>
      <nav class="tabs center-align ">
        {props.tabs.map((t) => (
          <a class={`tab vertical max ${ active() === t.key ? "active" : "background" }`}
            onClick={() => setActive(t.key)}
          >
            <i>{t.icon}</i>
            {t.label}
          </a>
        ))}
      </nav>

      <div class="page active">
        {props.tabs.map((t) => (
          <Show when={active() === t.key}>
            <ComponentWrapper tab={t}>
              <t.component />
            </ComponentWrapper>
          </Show>
        ))}
      </div>
    </div>
  );
}


interface ComponentWrapperProps {
  children: any;
  tab: Tab;
}

export default function ComponentWrapper(props: ComponentWrapperProps) {
  const resolved = children(() => props.children);
  return (
    <article>
      <header>
        <nav>
          <i>{props.tab.icon}</i>
          <h2 class="max">{props.tab.label}</h2>
        </nav>
      </header>
      {resolved()}
    </article>
  )
}