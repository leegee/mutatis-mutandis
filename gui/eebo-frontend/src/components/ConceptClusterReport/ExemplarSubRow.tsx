import { createSignal, createEffect, onCleanup, createMemo, Show } from "solid-js";
import { getWindow } from "../../services/windowCache";
import { controlsActions } from "../../state/controls.actions";
import type { ResolvedEvent } from "./ConceptClusters";

export default function ExemplarSubRow(props: { event: ResolvedEvent }) {
    let rowRef: HTMLTableRowElement | undefined;
    const [visible, setVisible] = createSignal(false);

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

    const windowText = createMemo(() => getWindow(props.event.event_id));

    return (
        <tr
            ref={rowRef}
            onClick={() => controlsActions.setSelectedEventIds(props.event.event_id)}
            class="bottom-padding surface-container-lowest"
        // style={{ background: controls.selectedEventId === props.event.event_id ? "var(--color-background-info)" : "transparent", }}
        >
            <td colspan="3">
                <div>
                    token_idx {props.event.token_idx} · {props.event.token}
                </div>

                <div>
                    <Show when={!visible()}>
                        &mdash;
                    </Show>

                    <Show when={visible() && !windowText()}>
                        <progress />
                    </Show>

                    <Show when={windowText()}>
                        <span innerHTML={windowText()!} />
                    </Show>
                </div>
            </td>
        </tr>
    );
}
