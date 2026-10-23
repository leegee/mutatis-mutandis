import { For, Show } from "solid-js";
import html from "solid-js/html";
import { pane } from "../store.js";
import { exportPane } from "../actions.js";

export function StatusBar(props) {
  const id = props.id;
  const p = () => pane(id);
  return html`
    <div class="status">
      <${Show} when=${() => p()?.status === "error"}>
        <div class="banner error" role="alert">${() => p()?.error}</div>
      <//>
      <${Show} when=${() => (p()?.warnings.length ?? 0) > 0}>
        <div class="banner warn"><${For} each=${() => p().warnings}>${(w) => html`<div>${w}</div>`}<//></div>
      <//>
      <${Show} when=${() => !!p()?.lastRequest}>
        <div class="meta">
          <span>${() => p().hits.length} result${() => (p().hits.length === 1 ? "" : "s")} · ${() => p().elapsedMs} ms</span>
          <span class="spacer"></span>
          <button class="btn small ghost" type="button" onClick=${() => exportPane(id, "json")}>Export JSON</button>
          <button class="btn small ghost" type="button" onClick=${() => exportPane(id, "csv")}>Export CSV</button>
        </div>
      <//>
    </div>`;
}
