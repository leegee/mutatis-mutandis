// src/services/tokenClustersService.ts

import type { Tier2Data } from "../components/NeighbourhoodBrowser";

export async function fetchTokenClusters(path: string): Promise<Tier2Data> {
    const url = `/json/${ path }`;
    console.log('[fetchTokenClusters] url', url)
    const res = await fetch(url);
    console.log('[fetchTokenClusters] got', url)
    if (!res.ok) throw new Error(`Failed to load ${ url }`);
    const json = await res.json();
    console.log('[fetchTokenClusters] json', json)
    return json;
}