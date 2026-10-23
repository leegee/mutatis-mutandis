import { For, Show } from "solid-js";
import html from "solid-js/html";
import { store } from "../store.js";
import { textStore, setTextStore } from "../textStore.js";
import { runTextSearch, exportTextResults } from "../textActions.js";
import { navigate, ROUTES } from "../router.js";
import { TextHitCard } from "./TextHitCard.js";

export function TextSearchView() {
  const field = (key) => () => textStore.form[key] ?? "";
  const setField = (key) => (e) => setTextStore("form", { [key]: e.currentTarget.value });

  return html`
    <header class="top">
      <h1>Passage Search</h1>
      <nav class="nav">
        <button type="button" class="btn small ghost"
                onClick=${() => navigate(ROUTES.search)}>Semantic</button>
        <button type="button" class="btn small on">Text</button>
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
    </header>

    <main class="panes single">
      <section class="pane">
        <form class="form" onSubmit=${(e) => { e.preventDefault(); runTextSearch(); }}>
          <label class="field wide"><span>Phrase or keywords</span>
            <input class="q" type="search" placeholder="e.g. white hair" autocomplete="off"
                   value=${field("query")}
                   onInput=${setField("query")} />
          </label>
          <div class="row">
            <label class="field num"><span>From year</span>
              <input type="text" inputmode="numeric" value=${field("startYear")} onInput=${setField("startYear")} /></label>
            <label class="field num"><span>To year</span>
              <input type="text" inputmode="numeric" value=${field("endYear")} onInput=${setField("endYear")} /></label>
            <label class="field num"><span>Limit</span>
              <input type="text" inputmode="numeric" value=${field("limit")} onInput=${setField("limit")} /></label>
            <label class="field num"><span>Window</span>
              <input type="text" inputmode="numeric" value=${field("windowTokens")} onInput=${setField("windowTokens")}
                     title="Tokens of context on each side of the match" /></label>
            <button class="btn primary" type="submit" disabled=${() => textStore.status === "loading"}>
              ${() => (textStore.status === "loading" ? "Searching…" : "Search")}</button>
          </div>
          <p class="muted" style="font-size:12px;margin:0">
            Exact phrase match over tokenised text in Postgres (case-insensitive).
            Window is how many tokens of context are shown around each hit.
          </p>
        </form>

        <${Show} when=${() => textStore.status === "error"}>
          <div class="banner error" role="alert">${() => textStore.error}</div>
        <//>
        <${Show} when=${() => textStore.warnings.length > 0}>
          <div class="banner warn"><${For} each=${() => textStore.warnings}>${(w) => html`<div>${w}</div>`}<//></div>
        <//>
        <${Show} when=${() => !!textStore.lastRequest}>
          <div class="meta">
            <span>${() => textStore.hits.length} result${() => (textStore.hits.length === 1 ? "" : "s")}
              · ${() => textStore.elapsedMs} ms</span>
            <span class="spacer"></span>
            <button class="btn small ghost" type="button" onClick=${() => exportTextResults("json")}>Export JSON</button>
            <button class="btn small ghost" type="button" onClick=${() => exportTextResults("csv")}>Export CSV</button>
          </div>
        <//>

        <ol class=${() => "hits" + (textStore.status === "loading" ? " dim" : "")}>
          <${For} each=${() => textStore.hits}>${(hit) => html`<${TextHitCard} hit=${hit} />`}<//>
        </ol>
        <${Show} when=${() => textStore.status === "ok" && textStore.hits.length === 0}>
          <p class="empty">No passages matched this phrase in the selected year range.</p>
        <//>
        <${Show} when=${() => textStore.status === "idle"}>
          <p class="empty">Enter a phrase and press Search.</p>
        <//>
      </section>
    </main>`;
}
