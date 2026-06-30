import { createResource, Show } from "solid-js";
import { queryEventById } from "../services/db";
import { fetchWindowBatch, type TextWindowItem } from "../services/tokenWindowBatchApi";
import { setWindowCache, getWindow } from "../services/windowCache";

interface TextWindowProps {
    eventid: string;
    style?: string;
}

export default function TextWindow(props: TextWindowProps) {
    const [windowText] = createResource(
        () => props.eventid,
        async (eventId): Promise<string | null> => {
            // already cached from a previous fetch elsewhere?
            const cached = getWindow(eventId);
            if (cached) return cached;

            const event = await queryEventById(eventId);
            if (!event || event.doc_id == null || event.token_idx == null) {
                return null;
            }

            const res = await fetchWindowBatch([{
                docId: event.doc_id,
                tokenIdx: event.token_idx,
            }]);

            const item: TextWindowItem | undefined = res.results[0];
            if (!item) return null;

            setWindowCache(event.event_id, item.content);
            return item.content;
        }
    );

    const style = () => "max-width: 100%;" + (props.style ?? '')

    return (
        <Show when={!windowText.loading} fallback={<progress />}>
            <Show when={windowText()} fallback={<progress />}>
                <q innerHTML={windowText()!} style={style()} />
            </Show>
        </Show>
    );
}
