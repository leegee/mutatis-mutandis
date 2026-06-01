import {
    createSignal,
    createMemo,
    Show,
    type Component,
    Match,
    Switch,
} from "solid-js";

import { createTokenWindowResource } from "../services/tokenWindowApi";

interface EventContextProps {
    docId: string;
    tokenIdx: number;
    open?: boolean;
}

const EventContext: Component<EventContextProps> = (props) => {
    const [open, setOpen] = createSignal(props.open ?? true);

    // Only produce a fetch source when open
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
        <>
            <button class="chip tiny" onClick={() => setOpen(v => !v)} >
                <Switch>
                    <Match when={!open()}>
                        <i>arrow_drop_down</i>
                        <span class="tooltip bottom">View Context</span>
                    </Match>
                    <Match when={open()}>
                        <i>arrow_drop_up</i>
                        <span class="tooltip bottom">Hide Context</span>
                    </Match>
                </Switch>
            </button>

            <Show when={open()}>
                <Show when={!window.loading} fallback={<progress class="light-green-text" />} >
                    <Show when={!window.error} fallback={<div class="error">Failed to load context</div>} >
                        <blockquote innerHTML={window() || ""} class="border" />
                    </Show>
                </Show>
            </Show>

        </>
    );
};

export default EventContext;