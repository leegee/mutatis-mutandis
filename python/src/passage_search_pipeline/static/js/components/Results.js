import { For, Show } from "solid-js";
import html from "solid-js/html";
import { pane } from "../store.js";
import { HitCard } from "./HitCard.js";

export function Results(props) {
  const id = props.id;
  const p = () => pane(id);
  return html`
    <div>
      <ol class=${() => "hits" + (p()?.status === "loading" ? " dim" : "")}>
        <${For} each=${() => p()?.hits ?? []}>${(hit) => html`<${HitCard} paneId=${id} hit=${hit} />`}<//>
      </ol>
      <${Show} when=${() => p()?.status === "ok" && p().hits.length === 0}>
        <p class="empty">No passages found for this query and year range.</p>
      <//>
      <${Show} when=${() => p()?.status === "idle"}>
        <p class="empty">Enter a query and press Search.</p>
      <//>
    </div>`;
}
