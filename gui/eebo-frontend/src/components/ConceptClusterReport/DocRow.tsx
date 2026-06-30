import { createSignal, createResource, createEffect, onCleanup, Show, For } from "solid-js";
import { fetchWindowBatch, type TextWindowItem } from "../../services/tokenWindowBatchApi";
import { getWindow, setWindowCache } from "../../services/windowCache";
import type { ResolvedEvent } from "./ConceptClusters";
import ExemplarSubRow from "./ExemplarSubRow";

export default function DocRow(props: {
    rank: number;
    doc_id: string;
    count: number;
    events: ResolvedEvent[];
}) {
    let rowRef: HTMLTableRowElement | undefined;
    const [visible, setVisible] = createSignal(false);

    // events are pre-resolved and already capped to MAX_EXEMPLARS_PER_DOC
    // by the parent, so only fetch window text for these few ids, and only
    // once this row is visible.
    const [resolved] = createResource(
        () => (visible() ? props.events : null),
        async (events): Promise<ResolvedEvent[]> => {
            if (!events || !events.length) return [];

            // skip ids whose window content is already cached
            const toFetch = events.filter((e) => !getWindow(e.event_id));

            if (toFetch.length) {
                const batch = toFetch.map((e) => ({
                    eventId: e.event_id,
                    docId: e.doc_id,
                    tokenIdx: e.token_idx,
                }));

                const res = await fetchWindowBatch(batch);

                res.results.forEach((r: TextWindowItem, idx: number) => {
                    setWindowCache(toFetch[idx].event_id, r.content);
                });
            }

            return events;
        }
    );

    createEffect(() => {
        if (!rowRef || visible()) return;
        const observer = new IntersectionObserver(
            (entries) => {
                if (entries[0].isIntersecting) {
                    setVisible(true);
                    observer.disconnect();
                }
            },
            { rootMargin: "100px" }
        );
        observer.observe(rowRef);
        onCleanup(() => observer.disconnect());
    });

    return (
        <>
            <tr ref={rowRef} class="surface-container">
                <td>{props.rank}</td>
                <td><strong>{props.doc_id}</strong></td>
                <td>{props.count.toLocaleString()}</td>
            </tr>
            <Show when={visible() && resolved.loading}>
                <tr class="surface-container-low">
                    <td colspan="3">
                        <progress />
                    </td>
                </tr>
            </Show>
            <For each={props.events}>
                {(ev) => <ExemplarSubRow event={ev} />}
            </For>
        </>
    );
}


