import type { Tier3GraphData } from "../types";

export async function loadDriftData(url: string): Promise<Tier3GraphData> {
    const res = await fetch('/api/' + url);

    if (!res.ok) {
        throw new Error(`Failed to load drift data: ${res.status}`);
    }

    const resp = await res.json();

    return resp.data as Tier3GraphData;
}
