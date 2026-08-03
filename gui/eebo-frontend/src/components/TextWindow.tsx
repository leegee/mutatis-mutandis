import { createResource, Show } from "solid-js";
import { queryEventById } from "../services/db";
import { fetchWindowBatch } from "../services/tokenWindowBatchApi";
import { setWindowCache, getWindowCacheStore } from "../services/windowCache";

type TextWindowProps =
    | {
        eventid: string;
        style?: string;
    }
    | {
        corpus: string;
        doc_id: string;
        token_idx: number;
        style?: string;
    };

export default function TextWindow(props: TextWindowProps) {
    const cache = getWindowCacheStore();

    const windowKey = () => {
        if ("eventid" in props) {
            return props.eventid;
        }

        return `${ props.doc_id }:${ props.token_idx }`;
    };

    const [windowText] = createResource(
        windowKey,
        async (key): Promise<string | null> => {
            const cached = cache[key];
            if (cached) return cached;

            let corpus: string | null = null;
            let docId: string | null = null;
            let tokenIdx: number | null = null;
            let cacheKey = key;

            if ("eventid" in props) {
                const event = await queryEventById(props.eventid);

                if (!event || event.doc_id == null || event.token_idx == null) {
                    return null;
                }

                corpus = event.corpus;
                docId = event.doc_id;
                tokenIdx = event.token_idx;
                cacheKey = event.event_id;
            } else {
                corpus = props.corpus;
                docId = props.doc_id;
                tokenIdx = props.token_idx;
            }

            const res = await fetchWindowBatch([
                {
                    corpus,
                    docId,
                    tokenIdx,
                },
            ]);

            const item = res.results[0];
            if (!item) return null;

            setWindowCache(cacheKey, item.content);
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