import { createResource, For, onCleanup, onMount, Show } from "solid-js";
import { queryEventById } from "../services/db";
import { showDocument } from "../services/documentApi";
import { fetchWindowBatch } from "../services/tokenWindowBatchApi";
import { controlsActions } from "../state/controls.actions";
import { controls } from "../state/controls.store";
import ExportSelectedEvents from "./ExportSelectedEvents";
import type { PointData } from "./ScatterPlot/types";

interface Props {
    onClose: () => void;
}

export default function SidebarMultiple(props: Props) {
    function handleKeyDown(e: KeyboardEvent) {
        if (e.key === "Escape") props.onClose();
    }

    onMount(() => window.document.body.addEventListener("keydown", handleKeyDown));
    onCleanup(() => window.document.body.removeEventListener("keydown", handleKeyDown));

    // stable selection
    const selectedEventIds = () => (controls.selectedEventIds ? Array.from(controls.selectedEventIds) : []);

    const [sidebarData] = createResource(selectedEventIds, async (ids) => {
        if (!ids.length) return null;

        // Get events by selection IDs
        const events = await Promise.all(ids.filter((id): id is string => id !== null).map((id) => queryEventById(id)));

        const cleanEvents = events.filter(Boolean);
        console.debug("[SidebarMultiple.sidebarData]", cleanEvents);

        // Get window-text snippets
        const windowRes = await fetchWindowBatch(
            cleanEvents.map((e: any) => ({
                docId: e.doc_id,
                tokenIdx: Number(e.token_idx),
            })),
        );

        const windowMap = new Map<string, any>();
        for (const w of windowRes.results) {
            windowMap.set(`${ w.docId }:${ Number(w.tokenIdx) }`, w);
        }

        // Group events for rendering
        const groupedMap = new Map<string, any[]>();

        for (const e of cleanEvents) {
            if (e) {
                // todo
                const list = groupedMap.get(e.doc_id) ?? [];
                list.push(e);
                groupedMap.set(e.doc_id, list);
            }
        }

        const grouped = [...groupedMap.entries()].map(([doc, evs]) => ({
            doc,
            events: evs.sort((a, b) => a.token_idx - b.token_idx),
        }));

        return {
            events: cleanEvents,
            windowMap,
            grouped,
        };
    });

    function handleOnClose() {
        controlsActions.setSelectedEventIds(null);
        if (props.onClose) props.onClose();
    }

    return (
        <Show when={controls.selectedEventIds}>
            {(selectedEventIds) => (
                <Show when={selectedEventIds().size > 0}>
                    <aside
                        class="surface-container padding scroll-parent  no-margin"
                        style={{ "max-width": "clamp(30rem, 30rem, 30vw)" }}
                    >
                        <header>
                            <nav>
                                <button type="button" class="small border no-margin no-padding circle" onClick={handleOnClose}>
                                    <i>close</i>
                                </button>
                                <h4 class="small max">
                                    {" "}
                                    {selectedEventIds().size} selected event
                                    {selectedEventIds().size !== 1 ? "s" : ""}
                                </h4>
                                <ExportSelectedEvents />
                            </nav>
                        </header>

                        <Show when={sidebarData.loading}>
                            <progress />
                        </Show>

                        <section style={{ overflow: "auto" }}>
                            <For each={sidebarData()?.grouped}>
                                {(group) => (
                                    <fieldset class="no-padding">
                                        <legend>
                                            <button type="button" class="chip" onClick={() => showDocument(group.doc)}>
                                                {group.doc} &mdash;{" "}
                                                <small>
                                                    {group.events[0].pub_year} (x {group.events.length})
                                                </small>
                                            </button>
                                        </legend>

                                        <span class="small left-padding medium-opacity">
                                            {group.events[0].author}
                                            &mdash;
                                            {group.events[0].pub_place}
                                        </span>

                                        <For each={group.events}>
                                            {(e: PointData) => {
                                                const data = sidebarData();
                                                const key = `${ e.doc_id }:${ Number(e.token_idx) }`;
                                                const w = data?.windowMap.get(key);

                                                return (
                                                    <>
                                                        <nav>
                                                            <h6 class="small bold left-margin no-padding max">
                                                                <q>{e.token}</q>
                                                            </h6>
                                                            <span title="Window ID/position">
                                                                {e.window_id}/{e.window_token_pos}
                                                            </span>
                                                        </nav>

                                                        <Show when={w} fallback={<progress />}>
                                                            <blockquote innerHTML={w.content} class="no-padding left-margin bottom-margin" />
                                                        </Show>
                                                    </>
                                                );
                                            }}
                                        </For>
                                    </fieldset>
                                )}
                            </For>
                        </section>
                    </aside>
                </Show>
            )}
        </Show>
    );
}
