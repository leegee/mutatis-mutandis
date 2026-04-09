// src/services/tokenClustersService.ts
export async function fetchTokenClusters(filename: string) {
    const res = await fetch(`/api/${filename}`);
    if (!res.ok) throw new Error(`Failed to fetch ${filename}`);
    return res.json();
}