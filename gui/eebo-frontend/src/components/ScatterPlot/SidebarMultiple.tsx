import { createMemo, createResource, For, Show } from "solid-js";
import { controls } from "../../state/controls.store";
import { queryEventById } from "../../services/db";
import { controlsActions } from "../../state/controls.actions";
import { createTokenWindowBatchResource } from "../../services/tokenWindowBatchApi";

export default function SidebarMultiple() {
    // selection snapshot
    const selectedIds = createMemo(() =>
        controls.selectedEventIds
            ? [...controls.selectedEventIds]
            : []
    );

    // load events
    const [events] = createResource(selectedIds, async (ids) => {
        if (!ids.length) return [];
        return Promise.all(ids.map((id) => queryEventById(id)));
    });

    // group by document
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

        return [...map.entries()].map(([doc, evs]) => ({
            doc,
            events: evs.sort((a, b) => a.token_idx - b.token_idx),
        }));
    });

    // window queries
    const windowQueries = createMemo(() => {
        const evs = events();
        if (!evs) return null;

        return evs
            .filter(Boolean)
            .map(e => ({
                eventId: String(e.event_id),
                docId: e.doc_id,
                tokenIdx: e.token_idx,
            }));
    });

    const [windows] = createTokenWindowBatchResource(windowQueries);

    // explicit window state (loading + map separated)
    const windowsState = createMemo(() => {
        const w = windows();

        const map = w?.results
            ? new Map(w.results.map(x => [x.eventId, x]))
            : null;

        console.log("[windowsState] loading:", windows.loading);
        console.log("[windowsState] keys:", map ? [...map.keys()] : []);
        console.log("[windowsState] sample eventId:", w?.results?.[0]?.eventId);

        return {
            loading: windows.loading,
            map
        };
    });

    const getWindow = (eventId: string) =>
        windowsState().map?.get(eventId);

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
                                <button
                                    class="small border"
                                    onClick={() =>
                                        controlsActions.setSelectedEventIds(null)
                                    }
                                >
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
                                            <span class="badge none">
                                                {group.events.length}
                                            </span>
                                        </button>

                                        <For each={group.events}>
                                            {(e) => {
                                                const w = getWindow(String(e.event_id));

                                                return (
                                                    <>
                                                        <div class="row left-margin">
                                                            <q>{e.token}</q>
                                                            <small>
                                                                {e.pub_year} · Token {e.token_idx}
                                                            </small>
                                                        </div>

                                                        <Show
                                                            when={w}
                                                            fallback={
                                                                <div class="loading">
                                                                    loading context…
                                                                </div>
                                                            }
                                                        >
                                                            <blockquote
                                                                innerHTML={w!.content}
                                                            />
                                                        </Show>
                                                    </>
                                                );
                                            }}
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