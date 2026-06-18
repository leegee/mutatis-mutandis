// src/components/TabsLayout.tsx

import { createSignal, Show } from "solid-js";

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
      <nav class="tabs center-align">
        {props.tabs.map((t) => (
          <a class={`tab ${ active() === t.key ? "active" : "" }`}
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
            <t.component tab={t} />
          </Show>
        ))}
      </div>
    </div>
  );
}
