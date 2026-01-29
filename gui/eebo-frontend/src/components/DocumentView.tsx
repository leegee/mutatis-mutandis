import { createSignal, createEffect, createMemo, Show, Match, Switch } from "solid-js";
import { documentXmlURL, fetchDocumentJson } from "../services/documentService";
import type { MyDocument } from "../types";

interface DocumentViewProps {
    docId: string | null; // null if nothing selected
}

export default function DocumentView(props: DocumentViewProps) {
    const docId = createMemo(() => props.docId);

    const [myDocument, setMyDocument] = createSignal<MyDocument | null>(null);
    const [format, setFormat] = createSignal<"json" | "xml">("json");
    const [loading, setLoading] = createSignal(false);
    const [error, setError] = createSignal<string | null>(null);
    const [xmlAvailable, setXmlAvailable] = createSignal<boolean | null>(null);

    // Fetch document whenever docId or format changes
    createEffect(() => {
        const id = docId();
        const fmt = format();

        if (!id) {
            setMyDocument(null);
            setError(null);
            setXmlAvailable(null);
            return;
        }

        setLoading(true);
        setError(null);

        if (fmt === "json") {
            fetchDocumentJson(id)
                .then((doc) => {
                    setMyDocument(doc);
                })
                .catch((err: any) => {
                    if (err?.status === 404) {
                        setError("Document not found.");
                    } else if (err?.status >= 500) {
                        setError("Server error while loading document.");
                    } else {
                        setError("Network error. Is the backend running?");
                    }
                    setMyDocument(null);
                })
                .finally(() => setLoading(false));
        }

        else if (fmt === "xml") {
            setMyDocument(null);
            setXmlAvailable(null);
            setLoading(true);

            fetch(documentXmlURL(id), { method: "HEAD" })
                .then(res => {
                    if (res.ok) {
                        setXmlAvailable(true);
                    } else if (res.status === 404) {
                        setXmlAvailable(false);
                        setError("XML version not available for this document.");
                    } else {
                        setXmlAvailable(false);
                        setError("Failed to load XML document.");
                    }
                })
                .catch(() => {
                    setXmlAvailable(false);
                    setError("Network error while checking XML.");
                })
                .finally(() => setLoading(false));
        }

        else {
            setMyDocument(null);
            setLoading(false);
        }
    });

    return (
        <Show when={docId()}>
            <Show when={!loading()} fallback={<div>Loading document...</div>}>
                <Show when={!error()} fallback={<div>Error: {error()}</div>}>
                    <article>
                        <header>
                            <p>
                                <strong>EEBO ID:</strong> {docId()} <br />
                                <strong>Author:</strong> {myDocument()?._source?.author ?? "-"} <br />
                                <strong>Year:</strong> {myDocument()?._source?.year ?? "-"} <br />
                                <strong>Place:</strong> {myDocument()?._source?.place ?? "-"} <br />
                                <strong>Publisher:</strong> {myDocument()?._source?.publisher ?? "-"}
                            </p>
                            <p>
                                <Show when={format() === "json"}>
                                    <button onClick={() => setFormat("xml")}>View XML</button>
                                </Show>
                                <Show when={format() === "xml"}>
                                    <button onClick={() => setFormat("json")}>View JSON</button>
                                </Show>
                            </p>
                            <h3>{myDocument()?._source?.title ?? "-"}</h3>
                        </header>

                        <hr />

                        <Switch>
                            <Match when={format() === "xml"}>
                                <iframe
                                    src={documentXmlURL(docId()!)}
                                    style={{ width: "100%", height: "60vh", border: "1px solid #ccc" }}
                                ></iframe>
                            </Match>
                            <Match when={myDocument() && format() === "json"}>
                                <div>{myDocument()?._source?.text ?? ""}</div>
                            </Match>
                        </Switch>
                    </article>
                </Show>
            </Show>
        </Show>
    );
}
