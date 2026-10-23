import { For, Show } from "solid-js";
import html from "solid-js/html";
import { pane } from "../store.js";
import { snippet } from "../helpers.js";
import { removeNegative, clearNegatives } from "../actions.js";

export function Negatives(props) {
  const id = props.id;
  const p = () => pane(id);
  return html`
    <${Show} when=${() => (p()?.negatives.length ?? 0) > 0}>
      <div class="negs">
        <span class="label">Excluding:</span>
        <${For} each=${() => p().negatives}>${(n) => html`
          <span class="chip" title=${n.snippet}>
            <span>${n.year} · ${snippet(n.snippet, 36)}</span>
            <button class="x" type="button" aria-label="Remove" onClick=${() => removeNegative(id, n.id)}>×</button>
          </span>`}<//>
        <button class="btn small ghost" type="button" onClick=${() => clearNegatives(id)}>Clear all</button>
      </div>
    <//>`;
}
