import type { Tier3GraphData } from "../types";

export async function loadDriftData(url: string): Promise<Tier3GraphData> {
    const res = await fetch('/api/' + url);

    if (!res.ok) {
        throw new Error(`Failed to load drift data from ${ url }: ${ res.status }`);
    }

    const resp = await res.json();

    if (!resp.data) {
        throw new Error(`Failed to find data key in response JSON: ${ resp }`);
    }

    return resp.data as Tier3GraphData;
}
