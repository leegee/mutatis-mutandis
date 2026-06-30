export interface ClusterLabelRequest {
    concept: string;

    points: {
        id: string;
        x: number;
        y: number;
    }[];

    rawText?: string; // optional fallback context
}

export interface Cluster2GroqRequest {
    concept: string;
    rawText?: string;
}


export async function cluster2groq(payload: Cluster2GroqRequest) {
    console.log("[cluster2groq] Enter")

    if (!payload.rawText) return;

    const seen = new Set<string>();
    const result: string[] = [];

    for (const text of payload.rawText.split(/\n+/)) {
        if (!text) continue;

        const normalised = text.toLowerCase().replace(/[\s\p{P}]+/gu, "");
        if (seen.has(normalised)) continue;

        seen.add(normalised);
        result.push(text);
    }

    payload.rawText = result.join("\n\n");

    console.log(`[cluster2groq]`, payload.rawText);

    const res = await fetch(`/api/groq`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
    });

    if (!res.ok) {
        throw new Error("Failed to load cluster label from Groq");
    }

    const json = await res.json();

    console.log(
        "[cluster2groq] RV",
        JSON.stringify(json, null, 2)
    );

    return json;
}
