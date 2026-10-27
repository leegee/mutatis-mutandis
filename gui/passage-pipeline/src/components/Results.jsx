import { For, Show } from "solid-js";
import { pane } from "../store.js";
import { HitCard } from "./HitCard.jsx";

export function Results(props) {
	const id = props.id;
	const p = () => pane(id);
	return (
		<div>
			<ol class={`hits${p()?.status === "loading" ? " dim" : ""}`}>
				<For each={p()?.hits ?? []}>
					{(hit) => <HitCard paneId={id} hit={hit} />}
				</For>
			</ol>
			<Show when={p()?.status === "ok" && p().hits.length === 0}>
				<p class="empty">No passages found for this query and year range.</p>
			</Show>
			<Show when={p()?.status === "idle"}>
				<p class="empty">Enter a query and press Search.</p>
			</Show>
		</div>
	);
}
