import { createMemo, Show } from "solid-js";

import { queryTerms, shortId, splitHighlight } from "../helpers.js";
import { textStore } from "../textStore.js";
import { copyText } from "../toast.js";

function TextHighlight(props) {
	const parts = createMemo(() => {
		const q = textStore.lastRequest?.query;
		return splitHighlight(props.text, q ? queryTerms(q) : []);
	});
	return <span>{parts().map((s) => (s.m ? <mark>{s.t}</mark> : s.t))}</span>;
}

export function TextHitCard(props) {
	const hit = props.hit;

	return (
		<li class="hit">
			<div class="hit-head">
				<span class="rank">{hit.rank}</span>
				<span class="year">{hit.pub_year || "—"}</span>
				<span class="tag">text match</span>
				<Show when={hit.match_start != null && hit.match_end != null}>
					<span class="muted" style="font-size:11px">
						match tokens {hit.match_start}:{hit.match_end}
					</span>
				</Show>
			</div>
			<p class="text">
				<TextHighlight text={hit.text ?? "[text unavailable]"} />
			</p>
			<div class="src">
				<span class="title">{hit.title ?? "(untitled)"}</span>
				<span>{hit.author ?? "(unknown author)"}</span>
				<span>
					{hit.corpus} / {hit.doc_id} · tokens {hit.token_start}:{hit.token_end}
				</span>
			</div>
			<div class="actions">
				<button
					class="btn small ghost"
					type="button"
					title={hit.passage_id}
					onClick={() => copyText(hit.passage_id)}
				>
					Copy ID {shortId(hit.passage_id)}
				</button>
			</div>
		</li>
	);
}
