import { type MyDocument } from "../types";

export async function fetchDocument(docId: string): Promise<MyDocument> {
    const url = `http://127.0.0.1:5000/documents/${docId}`;
    console.log(url);
    const res = await fetch(url);
    if (!res.ok) {
        throw new Error(`Failed to load document ${docId}`);
    }
    return res.json();
}