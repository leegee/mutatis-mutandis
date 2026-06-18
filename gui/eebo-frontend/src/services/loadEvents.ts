// src/services/loadEvents.ts

const URL = '/json/tier2_5_d3.json';

import type { SemanticEvent } from "../types";

export async function loadEvents(): Promise<SemanticEvent[]> {
    const res = await fetch(URL);

    if (!res.ok) {
        throw new Error("Failed to load semantic events");
    }

    const json = await res.json();
    const events: SemanticEvent[] = [];

    for (const conceptName of Object.keys(json.concepts)) {
        const concept = json.concepts[conceptName];

        for (const sliceId of Object.keys(concept.slices)) {
            const slice = concept.slices[sliceId];

            for (const inst of slice.instances) {
                if (!inst.xy) continue;

                events.push({
                    id: String(inst.vector_id),
                    vector_id: String(inst.vector_id),
                    token: String(inst.token),
                    doc_id: String(inst.doc_id),
                    filepath: String(inst.filepath),
                    concept: String(conceptName),

                    x: inst.xy.x,
                    y: inst.xy.y,

                    neighbours: inst.neighbours ?? []
                });
            }
        }
    }

    return events;
}