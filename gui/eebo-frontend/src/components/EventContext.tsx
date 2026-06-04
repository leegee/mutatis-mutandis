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
    label?: string | null;
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

    const [window, { refetch }] = createTokenWindowResource(source);

    return (
        <>
            <div class="row">
                <Show when={props.label}>
                    <h4 class="max">{props.label}</h4>
                </Show>
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
            </div>

            <Show when={open()}>
                <Show when={!window.loading} fallback={<progress class="light-green-text" />} >
                    <Show when={!window.error} fallback={
                        <aside class="error-container border padding">Failed to load context.
                            <button class="chip tiny no-border" onClick={refetch}>
                                <i>refresh</i>
                            </button>
                        </aside>
                    } >
                        <blockquote innerHTML={window() || ""} class="border" />
                    </Show>
                </Show>
            </Show>

        </>
    );
};

export default EventContext;