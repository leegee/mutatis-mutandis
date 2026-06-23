// src/lib/toast/ToastHost.tsx

import { For } from "solid-js";
import { removeToast, toasts } from "../state/toast.store";
import { Portal } from "solid-js/web";

import "./ToastHost.css";

export function ToastHost() {
  return (
    <Portal>
      <div class="toast-list">
        <For each={toasts()}>
          {(t) => (
            <div class={`toast toast--${ t.type }`}>
              <nav>
                <span class="max">
                  {t.message}
                </span>
                <button class="chip round small no-border" onClick={() => removeToast(t.id)}>
                  <i>close</i>
                </button>
              </nav>
            </div>
          )}
        </For>
      </div>
    </Portal>
  );
}
