import { createMemo } from "solid-js";
import html from "solid-js/html";
import { queryTerms, splitHighlight } from "../helpers.js";
import { pane } from "../store.js";

export function Highlight(props) {
	const parts = createMemo(() => {
		const q = pane(props.paneId)?.lastRequest?.query;
		return splitHighlight(props.text, q ? queryTerms(q) : []);
	});
	return html`<span>${() => parts().map((s) => (s.m ? html`<mark>${s.t}</mark>` : s.t))}</span>`;
}
