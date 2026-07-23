import { createResource, Show } from "solid-js";
import { queryEventById } from "../services/db";
import { fetchWindowBatch } from "../services/tokenWindowBatchApi";
import { setWindowCache, getWindowCacheStore } from "../services/windowCache";

interface TextWindowProps {
    eventid: string;
    style?: string;
}

export default function TextWindow(props: TextWindowProps) {
    const cache = getWindowCacheStore();

    const [windowText] = createResource(
        () => [
            props.eventid,
            cache[props.eventid],
        ],
        async ([eventId, cached]): Promise<string | null> => {
            if (cached) return cached;

            const event = await queryEventById(eventId);
            if (!event || event.doc_id == null || event.token_idx == null) {
                return null;
            }

            const res = await fetchWindowBatch([{
                docId: event.doc_id,
                tokenIdx: event.token_idx,
            }]);

            const item = res.results[0];
            if (!item) return null;

            setWindowCache(event.event_id, item.content);
            return item.content;
        }
    );

    const style = () => "max-width: 100%;" + (props.style ?? "");

    return (
        <Show when={!windowText.loading} fallback={<progress />}>
            <Show when={windowText()} fallback={<progress />}>
                <q innerHTML={windowText()!} style={style()} />
            </Show>
        </Show>
    );
}