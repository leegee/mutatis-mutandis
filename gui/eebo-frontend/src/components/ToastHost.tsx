// src/lib/toast/ToastHost.tsx

import { For } from "solid-js";
import { toasts } from "../state/toast.store";
import { Portal } from "solid-js/web";

import "./ToastHost.css";

export function ToastHost() {
  return (
    <Portal>
      <div class="toast-list">
        <For each={toasts()}>
          {(t) => (
            <div class={`toast toast--${ t.type }`}>
              {t.message}
            </div>
          )}
        </For>
      </div>
    </Portal>
  );
}
