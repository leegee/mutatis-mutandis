export const now = (): string => new Date().toISOString();

export const id = (): string => crypto.randomUUID();
