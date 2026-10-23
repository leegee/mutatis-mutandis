import { createEffect, onMount, Show } from "solid-js";
import html from "solid-js/html";
import { CONFIG } from "../config.js";
import { store } from "../store.js";
import { checkApi } from "../actions.js";
import { toast } from "../toast.js";
import { route, match, ROUTES } from "../router.js";
import { SearchView } from "./SearchView.js";
import { NotFound } from "./NotFound.js";

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

  // Treat / and /search as the same view for now; add more routes later.
  const isSearch = () => match(ROUTES.home) || match(ROUTES.search);

  return html`
    <div class="shell">
      <${Show} when=${isSearch} fallback=${html`<${NotFound} />`}>
        <${SearchView} />
      <//>
      <div class=${() => "toast" + (toast() ? " show" : "")} role="status">${toast}</div>
    </div>`;
}
