import { createEffect, For, onMount, Show } from "solid-js";
import html from "solid-js/html";
import { CONFIG } from "../config.js";
import { store } from "../store.js";
import { checkApi, addPane } from "../actions.js";
import { toast } from "../toast.js";
import { Pane } from "./Pane.js";

export function App() {
  onMount(checkApi);

  // Persist the form state (not the results) across reloads.
  createEffect(() => {
    const snap = store.panes.map((p) => ({
      label: p.label,
      source: { ...p.source },
      form: { ...p.form },
      negatives: p.negatives.map((n) => ({ ...n })),
    }));
    try {
      localStorage.setItem(CONFIG.storageKey, JSON.stringify(snap));
    } catch {
      /* ignore */
    }
  });

  const apiLabel = () =>
    store.apiOk === true
      ? "API connected"
      : store.apiOk === false
        ? "API unreachable"
        : "Checking API…";

  return html`
    <div class="shell">
      <header class="top">
        <h1>Passage Search</h1>
        <span class=${() => "api" + (store.apiOk === true ? " ok" : store.apiOk === false ? " bad" : "")}
              title=${() => store.apiError}>${apiLabel}</span>
        <span class="spacer"></span>
        <${Show} when=${() => CONFIG.maxPanes > 1}>
          <button class="btn small" type="button" disabled=${() => store.panes.length >= CONFIG.maxPanes}
                  onClick=${() => addPane()}>+ Add query</button>
        <//>
      </header>
      <main class=${() => "panes" + (store.panes.length === 1 ? " single" : "")}
            style=${() => `--cols:${store.panes.length}`}>
        <${For} each=${() => store.panes}>${(p) => html`<${Pane} id=${p.id} />`}<//>
      </main>
      <div class=${() => "toast" + (toast() ? " show" : "")} role="status">${toast}</div>
    </div>`;
}
