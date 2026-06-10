import { createMemo, createResource, For, Show } from "solid-js";
import { controls } from "../../state/controls.store";
import { queryEventById } from "../../services/db";
import { controlsActions } from "../../state/controls.actions";

export default function SidebarMultiple() {

    // selection to stable array snapshot
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

        // stable ordering: document to token_idx
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
                    <aside id="sidebar_container" class="surface-container-high padding">
                        <header>
                            <nav>
                                <h2>
                                    {selectedEventIds().size} selected events
                                </h2>
                                <button class="small border" onClick={() => controlsActions.setSelectedEventIds(null)}>
                                    <i>close</i>
                                </button>
                            </nav>
                        </header>

                        <section class="large-height scroll surface">
                            <For each={grouped()}>
                                {(group) => (
                                    <section class="doc-group">
                                        <button class="chip">
                                            <span class="large-text">
                                                {group.doc}
                                            </span>
                                            <span class="badge none">{group.events.length}</span>
                                        </button>

                                        <For each={group.events}>
                                            {(e) => (
                                                <div class="row left-margin">
                                                    <q>{e.token}</q>
                                                    <small>
                                                        {e.pub_year} · Token {e.token_idx}
                                                    </small>
                                                </div>
                                            )}
                                        </For>

                                    </section>
                                )}
                            </For>
                        </section>

                    </aside>
                </Show>
            )}
        </Show>
    );
}