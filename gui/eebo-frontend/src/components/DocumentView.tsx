import { createSignal, createEffect, Show } from "solid-js";
import { fetchDocument } from "../services/documentService";
import type { MyDocument } from "../types";

interface DocumentViewProps {
    docId: string | null; // null if nothing selected
}

export default function DocumentView(props: DocumentViewProps) {
    const [myDocument, setMyDocument] = createSignal<MyDocument | null>(null);
    const [loading, setLoading] = createSignal(false);
    const [error, setError] = createSignal<string | null>(null);

    createEffect(() => {
        const id = props.docId;
        console.log("props.docId", id);

        if (!id) {
            setMyDocument(null);
            return;
        }

        setLoading(true);
        setError(null);

        console.log("Fetching doc", id);
        fetchDocument(id)
            .then((doc) => {
                setMyDocument(doc);
                console.log("fetched", doc);
            })
            .catch((err) => setError(err.message))
            .finally(() => setLoading(false));
    });

    return (
        <Show when={props.docId}>
            <Show when={!loading()} fallback={<div>Loading document...</div>}>
                <Show when={!error()} fallback={<div>Error: {error()}</div>}>
                    <Show when={myDocument()}>
                        {(doc) => {
                            const { _source } = doc();
                            return (
                                <article>
                                    <header>
                                        <h2>{_source.title}</h2>
                                    </header>
                                    <p>
                                        <strong>Author:</strong> {_source.author} <br />
                                        <strong>Year:</strong> {_source.year} <br />
                                        <strong>Place:</strong> {_source.place} <br />
                                        <strong>Publisher:</strong> {_source.publisher}
                                    </p>
                                    <hr />
                                    <div>{_source.text}</div>
                                </article>
                            );
                        }}
                    </Show>
                </Show>
            </Show>
        </Show>
    );
}
