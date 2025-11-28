export const htmlDir = "./texts";

export const normalisePhiloId = (id) => id.replace(/\s+/g, "_");
export const htmlFile = (philoDiv1Id) => `${htmlDir}/${normalisePhiloId(philoDiv1Id)}.html`;
export const jsonFile = (philoDiv1Id) => `${htmlDir}/${normalisePhiloId(philoDiv1Id)}.json`;
