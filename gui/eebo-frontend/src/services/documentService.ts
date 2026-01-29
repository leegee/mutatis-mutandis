import { type MyDocument } from "../types";

export const documentURL = (docId: string) => `http://127.0.0.1:5000/documents/${docId}`;

export const documentXmlURL = (docId: string) => documentURL(docId) + '/xml';

export async function fetchDocumentJson(docId: string): Promise<MyDocument> {
    const res = await fetch(documentURL(docId));

    if (!res.ok) {
        const error: any = new Error(`Failed to load JSON document ${docId}`);
        error.status = res.status;
        throw error;
    }

    return res.json();
}

export async function fetchDocumentXml(docId: string): Promise<string> {
    const res = await fetch(documentXmlURL(docId), {
        headers: { Accept: "application/xml" },
    });

    if (!res.ok) {
        const error: any = new Error(`Failed to load XML document ${docId}`);
        error.status = res.status;
        throw error;
    }

    return res.text();
}
