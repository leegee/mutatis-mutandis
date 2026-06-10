import { createMemo, createResource, For, Show } from "solid-js";
import { controls } from "../../state/controls.store";
import { queryEventById } from "../../services/db";
import { controlsActions } from "../../state/controls.actions";

export default function SidebarMultiple() {

    // selection → stable array snapshot
    const selectedIds = createMemo(() => controls.selectedEventIds ? [...controls.selectedEventIds] : []);

    // load events in parallel
    const [events] = createResource(selectedIds, async (ids) => {
        if (!ids.length) return [];

        return Promise.all(
            ids.map((id) => queryEventById(id))
        );
    });

    // group by document (corpus-first mode)
    const grouped = createMemo(() => {
        const evs = events();
        if (!evs) return [];

        const map = new Map<string, any[]>();

        for (const e of evs) {
            if (!e) continue;
            const list = map.get(e.doc_id) ?? [];
            list.push(e);
            map.set(e.doc_id, list);
        }

        // stable ordering: document → token_idx
        return [...map.entries()]
            .map(([doc, evs]) => ({
                doc,
                events: evs.sort((a, b) => a.token_idx - b.token_idx)
            }));
    });

    return (
        <Show when={controls.selectedEventIds}>
            {(selectedEventIds) => (

                <Show when={selectedEventIds().size > 1}>
                    <aside id="sidebar_container" class="surface-container-high">

                        {/* HEADER */}
                        <header>
                            <h2>
                                {selectedEventIds().size} selected events
                            </h2>

                            <button onClick={() => controlsActions.setSelectedEventIds(null)}>
                                clear
                            </button>
                        </header>

                        {/* GROUPED BODY */}
                        <section>
                            <For each={grouped()}>
                                {(group) => (
                                    <div class="doc-group">

                                        <h3>
                                            {group.doc} ({group.events.length})
                                        </h3>

                                        <For each={group.events}>
                                            {(e) => (
                                                <div class="event-row">
                                                    <div>{e.token}</div>
                                                    <small>
                                                        {e.pub_year} · idx {e.token_idx}
                                                    </small>
                                                </div>
                                            )}
                                        </For>

                                    </div>
                                )}
                            </For>
                        </section>

                    </aside>
                </Show>
            )}
        </Show>
    );
}