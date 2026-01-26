import { createSignal, createEffect } from "solid-js";
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
        console.log('props.docId', props.docId);
        const id = props.docId;
        if (!id) {
            setMyDocument(null);
            return;
        }

        setLoading(true);
        setError(null);

        console.log('Fetching doc', id);
        fetchDocument(id)
            .then((doc) => {
                setMyDocument(doc);
                console.log('fetched', doc);
            })
            .catch((err) => setError(err.message))
            .finally(() => setLoading(false))
    });

    if (!props.docId) {
        return <div>Select a document to view</div>;
    }

    if (loading()) {
        return <div>Loading document...</div>;
    }

    if (error()) {
        return <div>Error: {error()}</div>;
    }

    const doc = myDocument();
    if (!doc) return null;

    const { _source } = doc;

    return (
        <div>
            <h2>{_source.title}</h2>
            <p>
                <strong>Author:</strong> {_source.author} <br />
                <strong>Year:</strong> {_source.year} <br />
                <strong>Place:</strong> {_source.place} <br />
                <strong>Publisher:</strong> {_source.publisher}
            </p>
            <hr />
            <div>{_source.text}</div>
        </div>
    );
}
