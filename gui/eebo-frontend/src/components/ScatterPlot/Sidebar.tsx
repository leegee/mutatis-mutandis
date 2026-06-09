import { createResource, Show } from "solid-js";
import { controls } from "../../state/controls.store";
import { queryEventById } from "../../services/db";
import { controlsActions } from "../../state/controls.actions";

async function fetchWindow(docId: string, tokenIdx: number) {
    const res = await fetch(`/api/window/${ docId }/${ tokenIdx }`);
    if (!res.ok) throw new Error("Failed to load window");
    return res.text();
}

export default function Sidebar() {
    const [event] = createResource(
        () => controls.selectedEventId,
        (id) => (id ? queryEventById(id) : null)
    );

    const [windowHtml] = createResource(
        event,
        (e) => {
            if (!e?.doc_id || e.token_idx == null) return null;
            return fetchWindow(e.doc_id, e.token_idx);
        }
    );

    return (
        <Show when={controls.selectedEventId}>
            <Show when={event()} fallback={null}>
                {(e) => (
                    <aside
                        id="sidebar_container"
                        class="min surface-container-high scroll medium-elevate border no-padding no-margin no-round"
                    >
                        <article>
                            <header>
                                <div class="row">
                                    <h2 class="max large">
                                        <q>{e().token}</q>
                                    </h2>
                                    <button class="link border" onClick={() => controlsActions.setSelectedEventId(null)} >
                                        <i>close</i>
                                    </button>
                                </div>
                            </header>

                            <section class="small-padding">
                                <div class="small-text">
                                    <div><b>Event ID:</b> {e().event_id}</div>
                                    <div><b>Doc:</b> {e().doc_id}</div>
                                    <div><b>Year:</b> {e().pub_year}</div>
                                    <div><b>Token Index:</b> {e().token_idx}</div>
                                </div>
                            </section>

                            <section class="small-padding">
                                <Show
                                    when={windowHtml()}
                                    fallback={<span>Loading context...</span>}
                                >
                                    {(html) => (
                                        <div innerHTML={html()} />
                                    )}
                                </Show>
                            </section>
                        </article>
                    </aside>
                )}
            </Show>
        </Show>
    );
}