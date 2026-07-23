import { createSignal, createEffect, onCleanup, createMemo, Show } from "solid-js";
import { getWindow } from "../../services/windowCache";
import { controlsActions } from "../../state/controls.actions";
import TextWindow from "../TextWindow";

interface Exemplar {
    event_id: string;
    token_idx: number;
    token?: string;
}

export default function ExemplarSubRow(props: { event: Exemplar }) {
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

    return (
        <tr ref={rowRef}
            onClick={() =>
                controlsActions.setSelectedEventIds(props.event.event_id)
            }
            class="bottom-padding surface-container-lowest"
        >
            <td colspan="7">
                <div>
                    <Show when={visible()}>
                        <span title={"token_idx " + props.event.token_idx}>
                            {props.event.token ? ` · ${ props.event.token }` : ""}
                        </span>

                        ...<TextWindow eventid={props.event.event_id} />...
                    </Show>
                </div>
            </td>
        </tr >
    );
}
