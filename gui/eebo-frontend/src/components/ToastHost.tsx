// src/lib/toast/ToastHost.tsx

import { For } from "solid-js";
import { toasts } from "../state/toast.store";

export function ToastHost() {
  return (
    <div class="toast-container top center">
      <For each={toasts}>
        {(t) => (
          <div class={`toast ${ t.type }`}>
            {t.message}
          </div>
        )}
      </For>
    </div>
  );
}