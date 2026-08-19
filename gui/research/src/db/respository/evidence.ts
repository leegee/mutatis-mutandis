import type { Evidence } from "~/domain/evidence";
import { getDatabase } from "../database";

export async function listEvidence(): Promise<Evidence[]> {
	return getDatabase().evidence.orderBy("createdAt").toArray();
}

export async function getEvidence(id: string): Promise<Evidence | undefined> {
	return getDatabase().evidence.get(id);
}

export async function createEvidence(evidence: Evidence): Promise<Evidence> {
	await getDatabase().evidence.add(evidence);

	return evidence;
}

export async function updateEvidence(
	evidence: Evidence,
	changes: Partial<Evidence>,
): Promise<Evidence> {
	const updated: Evidence = {
		...evidence,
		...changes,
	};

	await getDatabase().evidence.put(updated);

	return updated;
}

export async function deleteEvidence(id: string): Promise<void> {
	await getDatabase().evidence.delete(id);
}
