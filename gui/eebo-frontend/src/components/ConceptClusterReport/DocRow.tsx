import { createSignal, createResource, createEffect, onCleanup, Show, For } from "solid-js";
import { fetchWindowBatch } from "../../services/tokenWindowBatchApi";
import { getWindow, setWindowCache } from "../../services/windowCache";
import ExemplarSubRow from "./ExemplarSubRow";
import { showDocument } from "../../services/documentApi";

import './DocRow.css';

interface Props {
    rank: number;
    doc_id: string;
    title: string | null;
    author: string | null;
    pub_year: number | null;
    count: number;

    exemplars: {
        event_id: string;
        doc_id: string;
        token_idx: number;
    }[];
}

export default function DocRow(props: Props) {
    let rowRef: HTMLTableRowElement | undefined;
    const [visible, setVisible] = createSignal(false);

    const [resolved] = createResource(
        () => (visible() ? props.exemplars : null),
        async (exemplars) => {
            if (!exemplars?.length) return [];

            const toFetch = exemplars.filter(e => !getWindow(e.event_id));

            if (toFetch.length) {
                const batch = toFetch.map(e => ({
                    eventId: e.event_id,
                    docId: e.doc_id,
                    tokenIdx: e.token_idx,
                }));

                const res = await fetchWindowBatch(batch);

                res.results.forEach((r, idx) => {
                    setWindowCache(toFetch[idx].event_id, r.content);
                });
            }

            return exemplars;
        }
    );

    createEffect(() => {
        if (!rowRef || visible()) return;

        const observer = new IntersectionObserver(entries => {
            if (entries[0].isIntersecting) {
                setVisible(true);
                observer.disconnect();
            }
        }, { rootMargin: "100px" });

        observer.observe(rowRef);
        onCleanup(() => observer.disconnect());
    });

    return (
        <>
            <tr ref={rowRef} class="surface-container-highest">
                <td>{props.rank}</td>
                <td>{props.count.toLocaleString()}</td>
                <td>
                    <a class="link" onClick={() => showDocument(props.doc_id)}>
                        {props.doc_id}
                    </a>
                </td>
                <td>{props.author}</td>
                <td>{props.pub_year}</td>
                <td>
                    <span class="td-title">{props.title}</span>
                    <span class="tooltip max bottom">{props.title}</span>
                </td>
            </tr>

            <Show when={visible() && resolved.loading}>
                <tr>
                    <td colspan="6">
                        <progress />
                    </td>
                </tr>
            </Show>

            <For each={props.exemplars}>
                {(ev) => <ExemplarSubRow event={ev} />}
            </For>
        </>
    );
}
