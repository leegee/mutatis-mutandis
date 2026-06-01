import {
    createSignal,
    createMemo,
    Show,
    type Component,
} from "solid-js";
import { createTokenWindowResource } from "../services/tokenWindowApi";

interface EventContextProps {
    docId: string;
    tokenIdx: number;
    open?: boolean;
}

const EventContext: Component<EventContextProps> = (props) => {
    const [open, setOpen] = createSignal(props.open ?? true);

    const source = createMemo(() =>
        open()
            ? {
                doc_id: props.docId,
                token_idx: props.tokenIdx,
            }
            : null
    );

    const [window] = createTokenWindowResource(source);

    return (
        <details
            open={props.open}
            onToggle={(e) => setOpen(e.currentTarget.open)}
        >
            <summary>Context</summary>

            <Show when={open()}>
                <Show when={!window.loading} fallback={<progress class="light-green-text" />}>
                    <Show when={!window.error} fallback={
                        <div class="error">Failed to load context</div>
                    }>
                        <blockquote>{window()}</blockquote>
                    </Show>
                </Show>
            </Show>
        </details>
    );
};

export default EventContext;
