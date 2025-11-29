export const htmlDir = "./texts";

export const normalisePhiloId = (id: string) => id.replace(/\s+/g, "_");
export const htmlFile = (philoDiv1Id: string) => `${htmlDir}/${normalisePhiloId(philoDiv1Id)}.html`;
export const jsonFile = (philoDiv1Id: string) => `${htmlDir}/${normalisePhiloId(philoDiv1Id)}.json`;
