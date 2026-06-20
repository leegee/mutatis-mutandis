// src/lib/toast/ToastHost.tsx

import { For } from "solid-js";
import { toasts } from "../state/toast.store";
import { Portal } from "solid-js/web";

export function ToastHost() {
  return (
    <Portal>
      <div class="top" style="z-index: 999; position: fixed; width: 40rem; left: -20rem; margin-left: 50%; margin-top: 3rem;">
        <For each={toasts}>
          {(t) => (
            <div class={`snaackbar ${ t.type }-container margin padding  active`}>
              {t.message}
            </div>
          )}
        </For>
      </div>
    </Portal>
  );
}