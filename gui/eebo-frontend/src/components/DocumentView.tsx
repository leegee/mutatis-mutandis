import { createSignal, createEffect, Show, Match, Switch } from "solid-js";
import { documentURL, documentXmlURL, fetchDocumentJson, fetchDocumentXml } from "../services/documentService";
import type { MyDocument } from "../types";

interface DocumentViewProps {
    docId: string | null; // null if nothing selected
}

export default function DocumentView(props: DocumentViewProps) {
    const [myDocument, setMyDocument] = createSignal<MyDocument | null>(null);
    const [format, setFormat] = createSignal<"json" | "xml">("json");
    const [xmlContent, setXmlContent] = createSignal<string | null>(null);
    const [loading, setLoading] = createSignal(false);
    const [error, setError] = createSignal<string | null>(null);

    createEffect(() => {
        const id = props.docId;
        const fmt = format();

        if (!id) {
            setMyDocument(null);
            setXmlContent(null);
            return;
        }

        setLoading(true);
        setError(null);

        if (fmt === "json") {
            fetchDocumentJson(id)
                .then((doc) => {
                    setMyDocument(doc);
                    setXmlContent(null);
                })
                .catch((err) => setError(err.message))
                .finally(() => setLoading(false));
        }
    });

    return (
        <Show when={props.docId}>
            <Show when={!loading() || format() === 'xml'} fallback={<div>Loading document...</div>}>
                <Show when={!error()} fallback={<div>Error: {error()}</div>}>
                    <Switch>
                        <Match when={format() === 'xml'}>
                            <iframe
                                src={documentXmlURL(props.docId!)}
                                style={{ width: "100%", height: "500px", border: "1px solid #ccc" }}
                            ></iframe>
                        </Match>
                        <Match when={myDocument() && format() === 'json'}>
                            {(doc) => {
                                const { _source } = myDocument()!;
                                return (
                                    <article>
                                        <header>
                                            <p>
                                                <Show when={format() === 'json'}>
                                                    <button onClick={() => setFormat("xml")}>View XML</button>
                                                </Show>
                                                <Show when={format() === 'xml'}>
                                                    <button onClick={() => setFormat("json")}>View Metadata</button>
                                                </Show>
                                            </p>
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
                        </Match>
                    </Switch>
                </Show>
            </Show>
        </Show >
    );
}
