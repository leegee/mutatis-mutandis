import { createResource, Show, For } from "solid-js";
import { controls } from "../../state/controls.store";
import { queryEventById } from "../../services/db";
import { controlsActions } from "../../state/controls.actions";
import { fetchWindowBatch } from "../../services/tokenWindowBatchApi";
import ExportSelectedEvents from "../ExportSelectedEvents";
import { showDocument } from "../../services/documentApi";

export default function SidebarMultiple() {
    // stable selection
    const selectedEventIds = () => controls.selectedEventIds ? Array.from(controls.selectedEventIds) : [];

    const [sidebarData] = createResource(selectedEventIds, async (ids) => {
        if (!ids.length) return null;

        // Get events by selection IDs
        const events = await Promise.all(
            ids.map((id) => queryEventById(id))
        );

        const cleanEvents = events.filter(Boolean);

        // Get window-text snippets
        const windowRes = await fetchWindowBatch(
            cleanEvents.map((e: any) => ({
                docId: e.doc_id,
                tokenIdx: Number(e.token_idx),
            }))
        );

        const windowMap = new Map<string, any>();
        for (const w of windowRes.results) {
            windowMap.set(`${ w.docId }:${ Number(w.tokenIdx) }`, w);
        }

        // Group events for rendering
        const groupedMap = new Map<string, any[]>();

        for (const e of cleanEvents) {
            if (e) { // todo
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

    return (
        <Show when={controls.selectedEventIds}>
            {(selectedEventIds) => (
                <Show when={selectedEventIds().size > 1}>
                    <aside class="surface-container padding scroll-parent  no-margin"
                        style={{ "max-width": "clamp(30rem, 30rem, 30vw)" }}>
                        <header>
                            <nav>
                                <button class="small border no-margin no-padding circle" onClick={() => controlsActions.setSelectedEventIds(null)} >
                                    <i>close</i>
                                </button>
                                <h2 class="max"> {selectedEventIds().size} selected events </h2>
                                <ExportSelectedEvents />
                            </nav>
                        </header>


                        <section style={{ overflow: "auto" }}>
                            <For each={sidebarData()?.grouped}>
                                {(group) => (
                                    <fieldset class="no-padding">
                                        <legend>
                                            <button class="chip" onClick={() => showDocument(group.doc)}>
                                                {group.doc}
                                                <span class="none"> (x {group.events.length}) </span>
                                            </button>
                                        </legend>

                                        <For each={group.events}>
                                            {(e) => {
                                                const data = sidebarData();
                                                const key = `${ e.doc_id }:${ Number(e.token_idx) }`;
                                                const w = data?.windowMap.get(key);

                                                return (
                                                    <>
                                                        <h6 class="small left-margin no-padding">
                                                            <q>{e.token}</q>
                                                            &nbsp;&mdash;
                                                            <small> {e.pub_year} </small>
                                                        </h6>

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