const URL = '/json/indexes/tier2_concept_neighbours.json';

export async function loadConceptNeighbours(): Promise<any> {
    const res = await fetch(URL);

    if (!res.ok) {
        throw new Error(`Failed to load semantic events: ${ res.status } ${ res.statusText }`);
    }

    return res.json() as Promise<any>;
}
