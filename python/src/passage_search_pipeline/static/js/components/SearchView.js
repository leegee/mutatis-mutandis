import { For, Show } from "solid-js";
import html from "solid-js/html";
import { CONFIG } from "../config.js";
import { store } from "../store.js";
import { addPane } from "../actions.js";
import { navigate, ROUTES } from "../router.js";
import { Pane } from "./Pane.js";

/**
 * The main multi-pane search UI. Rendered for `#/` and `#/search`.
 * Extracted so App can switch routes without duplicating layout.
 */
export function SearchView() {
  return html`
    <header class="top">
      <h1>Passage Search</h1>
      <nav class="nav">
        <button type="button" class="btn small on">Semantic</button>
        <button type="button" class="btn small ghost"
                onClick=${() => navigate(ROUTES.text)}>Text</button>
      </nav>
      <span class=${() => "api" + (store.apiOk === true ? " ok" : store.apiOk === false ? " bad" : "")}
            title=${() => store.apiError}>${() =>
        store.apiOk === true
          ? "API connected"
          : store.apiOk === false
            ? "API unreachable"
            : "Checking API…"
      }</span>
      <span class="spacer"></span>
      <${Show} when=${() => CONFIG.maxPanes > 1}>
        <button class="btn small" type="button" disabled=${() => store.panes.length >= CONFIG.maxPanes}
                onClick=${() => addPane()}>+ Add query</button>
      <//>
    </header>
    <main class=${() => "panes" + (store.panes.length === 1 ? " single" : "")}
          style=${() => `--cols:${store.panes.length}`}>
      <${For} each=${() => store.panes}>${(p) => html`<${Pane} id=${p.id} />`}<//>
    </main>`;
}
