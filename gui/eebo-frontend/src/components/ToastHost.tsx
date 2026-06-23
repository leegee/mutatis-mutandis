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
          {(thisToast) => (
            <div class={`toast toast--${ thisToast.type }`}
              style={{
                animation: `toast-out ${ thisToast.durationMs }ms ease-out`,
                "animation-delay": `${ thisToast.durationMs - 2_000 }ms`
              }}
              onAnimationEnd={(e) => {
                if (e.animationName === "toast-out") {
                  removeToast(thisToast.id);
                }
              }}
            >
              <nav>
                <span class="max">
                  {thisToast.message}
                </span>
                <button class="chip round small no-border" onClick={() => removeToast(thisToast.id)}>
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
