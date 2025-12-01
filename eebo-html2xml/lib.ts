export const htmlDir = "./texts";

export const normalisePhiloId = (id: string) => id.replace(/\s+/g, "_");
export const htmlFile = (philoDiv1Id: string) => `${htmlDir}/${normalisePhiloId(philoDiv1Id)}.html`;
export const jsonFile = (philoDiv1Id: string) => `${htmlDir}/${normalisePhiloId(philoDiv1Id)}.json`;

export async function fetchJSON(url: string): Promise<any> {
    try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error("Fetch error:", url, err);
        return null;
    }
}
