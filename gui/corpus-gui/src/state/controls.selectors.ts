export async function getConcepts(): Promise<string[]> {
	return ["WHITE"];
}

export function selectIds<T>(rows: T[], predicate: (row: T) => boolean, getId: (row: T) => string): Set<string> {
	return new Set(rows.filter(predicate).map(getId));
}
