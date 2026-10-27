import { Show } from "solid-js";
import html from "solid-js/html";
import { removePane } from "../actions.js";
import { pane, store } from "../store.js";
import { Histogram } from "./Histogram.js";
import { Negatives } from "./Negatives.js";
import { Results } from "./Results.js";
import { SearchForm } from "./SearchForm.js";
import { StatusBar } from "./StatusBar.js";

export function Pane(props) {
	const id = props.id;
	return html`
    <section class="pane">
      <div class="pane-head">
        <span class="badge">${() => pane(id)?.label}</span>
        <span class="spacer"></span>
        <${Show} when=${() => store.panes.length > 1}>
          <button class="btn small ghost" type="button" onClick=${() => removePane(id)}>Remove</button>
        <//>
      </div>
      <${SearchForm} id=${id} />
      <${Negatives} id=${id} />
      <${StatusBar} id=${id} />
      <${Histogram} id=${id} />
      <${Results} id=${id} />
    </section>`;
}
