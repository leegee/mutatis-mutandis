// src/services/tokenClustersService.ts
import type { Dataset } from "../types";

export async function fetchTokenClusters(filename: string): Promise<Dataset> {
    const url = `/json/${ filename }`;
    console.log('[fetchTokenClusters] url', url)
    const res = await fetch(url);
    console.log('[fetchTokenClusters] got', url)
    if (!res.ok) throw new Error(`Failed to fetch ${ filename }`);
    const json = await res.json();
    console.log('[fetchTokenClusters] json', json)
    return json;
}