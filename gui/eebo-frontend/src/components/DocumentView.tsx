import { createSignal, createEffect, Show, Match, Switch } from "solid-js";
import { documentXmlURL, fetchDocumentJson } from "../services/documentService";
import type { MyDocument } from "../types";

interface DocumentViewProps {
    docId: string | null; // null if nothing selected
}

export default function DocumentView(props: DocumentViewProps) {
    const [myDocument, setMyDocument] = createSignal<MyDocument | null>(null);
    const [format, setFormat] = createSignal<"json" | "xml">("json");
    const [loading, setLoading] = createSignal(false);
    const [error, setError] = createSignal<string | null>(null);

    createEffect(() => {
        const id = props.docId;
        const fmt = format();

        if (!id) {
            setMyDocument(null);
            return;
        }

        setLoading(true);
        setError(null);

        if (fmt === "json") {
            fetchDocumentJson(id)
                .then((doc) => {
                    setMyDocument(doc);
                })
                .catch((err) => setError(err.message))
                .finally(() => setLoading(false));
        }
    });

    return (
        <Show when={props.docId} fallback={<div>No document selected</div>}>
            <Show when={!loading() || format() === "xml"} fallback={<div>Loading document...</div>}>
                <Show when={!error()} fallback={<div>Error: {error()}</div>}>
                    <article>
                        <header>
                            <p>
                                <strong>EEBO ID:</strong> {props.docId} <br />
                                <strong>Author:</strong> {myDocument()?._source.author} <br />
                                <strong>Year:</strong> {myDocument()?._source.year} <br />
                                <strong>Place:</strong> {myDocument()?._source.place} <br />
                                <strong>Publisher:</strong> {myDocument()?._source.publisher}
                            </p>
                            <p>
                                <Show when={format() === "json"}>
                                    <button onClick={() => setFormat("xml")}>View XML</button>
                                </Show>
                                <Show when={format() === "xml"}>
                                    <button onClick={() => setFormat("json")}>View JSON</button>
                                </Show>
                            </p>
                            <h3>{myDocument()?._source.title}</h3>
                        </header>

                        <hr />

                        <Switch>
                            <Match when={format() === "xml"}>
                                <iframe
                                    src={documentXmlURL(props.docId!)}
                                    style={{ width: "100%", height: "60vh", border: "1px solid #ccc" }}
                                ></iframe>
                            </Match>
                            <Match when={myDocument() && format() === "json"}>
                                <div>{myDocument()?._source.text}</div>
                            </Match>
                        </Switch>
                    </article>
                </Show>
            </Show>
        </Show>
    );
}
