import type { Tier2Data } from "../components/ConceptGraph/ConceptGraph.types";

const URL = '/json/indexes/tier2_concept_neighbours.json';

export async function loadConceptNeighbours(): Promise<Tier2Data> {
    const res = await fetch(URL);

    if (!res.ok) {
        throw new Error(`Failed to load semantic events: ${ res.status } ${ res.statusText }`);
    }

    return res.json() as Promise<Tier2Data>;
}
