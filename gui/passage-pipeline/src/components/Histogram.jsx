import { createMemo, For, Show } from "solid-js";

import { narrowYears } from "../actions.js";
import { computeBins } from "../helpers.js";
import { pane } from "../store.js";

export function Histogram(props) {
	const id = props.id;
	const bins = createMemo(() => computeBins(pane(id)?.hits ?? []));
	const maxCount = createMemo(() => Math.max(1, ...bins().map((b) => b.count)));
	return (
		<Show when={bins().length > 1}>
			<div class="hist">
				<div class="bars">
					<For each={bins()}>
						{(b) => (
							<button
								class="bar"
								type="button"
								title={`${b.start}–${b.end}: ${b.count} hit${b.count === 1 ? "" : "s"}`}
								onClick={() => narrowYears(id, b.start, b.end)}
							>
								<span
									class="fill"
									style={{
										height: `${Math.round((100 * b.count) / maxCount())}%`,
									}}
								/>
							</button>
						)}
					</For>
				</div>
				<div class="axis">
					<span>{bins()[0]?.start}</span>
					<span class="hint">click a bar to narrow the years</span>
					<span>{bins()[bins().length - 1]?.end}</span>
				</div>
			</div>
		</Show>
	);
}
