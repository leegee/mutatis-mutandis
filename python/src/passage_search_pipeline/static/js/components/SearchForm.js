import { For, Show } from "solid-js";
import html from "solid-js/html";
import { store, pane, update } from "../store.js";
import { shortId } from "../helpers.js";
import { runSearch, clearVectorSource } from "../actions.js";

export function SearchForm(props) {
  const id = props.id;
  const p = () => pane(id);
  const field = (key) => () => p()?.form[key] ?? "";
  const setField = (key) => (e) => update(id, "form", { [key]: e.currentTarget.value });

  return html`
    <form class="form" onSubmit=${(e) => { e.preventDefault(); runSearch(id); }}>
      <${Show} when=${() => p()?.source.mode === "vector"}
        fallback=${html`
          <label class="field wide"><span>Query</span>
            <input class="q" type="search" placeholder="e.g. white hair" autocomplete="off"
                   value=${() => p()?.source.text ?? ""}
                   onInput=${(e) => update(id, "source", { text: e.currentTarget.value })} />
          </label>`}>
        <div class="field wide"><span>Passages similar to</span>
          <div class="vecchip">
            <code>${() => shortId(p()?.source.vectorId)}</code>
            <span class="muted">${() => p()?.source.vectorLabel}</span>
            <button class="btn small" type="button" onClick=${() => clearVectorSource(id)}>Use text query</button>
          </div>
        </div>
      <//>

      <div class="row">
        <label class="field"><span>Representation</span>
          <select onChange=${setField("representation")}>
            <${For} each=${() => store.representations}>${(r) => html`
              <option value=${r} selected=${() => p()?.form.representation === r}>${r}</option>`}<//>
          </select>
        </label>
        <label class="field num"><span>From year</span>
          <input type="text" inputmode="numeric" value=${field("startYear")} onInput=${setField("startYear")} /></label>
        <label class="field num"><span>To year</span>
          <input type="text" inputmode="numeric" value=${field("endYear")} onInput=${setField("endYear")} /></label>
        <label class="field num"><span>Limit</span>
          <input type="text" inputmode="numeric" value=${field("limit")} onInput=${setField("limit")} /></label>
        <button class="btn primary" type="submit" disabled=${() => p()?.status === "loading"}>
          ${() => (p()?.status === "loading" ? "Searching…" : "Search")}</button>
      </div>

      <details class="adv">
        <summary>Re-ranking</summary>
        <div class="row">
          <label class="field num"><span>Beta</span>
            <input type="text" inputmode="decimal" value=${field("beta")} onInput=${setField("beta")} /></label>
          <label class="field num"><span>Overfetch</span>
            <input type="text" inputmode="numeric" value=${field("overfetch")} onInput=${setField("overfetch")} /></label>
          <span class="muted" style="font-size:12px;max-width:340px">
            Beta weighs the excluded passages against the query; overfetch is how many times
            <em>limit</em> candidates are drawn per table before re-ranking.</span>
        </div>
      </details>
    </form>`;
}
