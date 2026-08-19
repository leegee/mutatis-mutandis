import type { Evidence } from "~/domain/evidence";
import { getDatabase } from "../database";
import { id, now } from "./utils";

export async function createEvidence(
	sourceId: string,
	observation: string,
	status: Evidence["status"] = "primary",
	entityIds: string[] = [],
	relationIds: string[] = [],
	quote?: string,
	notes?: string,
): Promise<Evidence> {
	const evidence: Evidence = {
		id: id(),
		sourceId,
		entityIds,
		relationIds,
		observation,
		status,
		...(quote ? { quote } : {}),
		...(notes ? { notes } : {}),
		createdAt: now(),
	};

	await getDatabase().evidence.add(evidence);

	return evidence;
}

export async function updateEvidence(
	evidence: Evidence,
	changes: Partial<Omit<Evidence, "id" | "createdAt">>,
): Promise<Evidence> {
	const updated: Evidence = {
		...evidence,
		...changes,
	};

	await getDatabase().evidence.put(updated);

	return updated;
}

export async function getEvidence(
	evidenceId: string,
): Promise<Evidence | undefined> {
	return getDatabase().evidence.get(evidenceId);
}

export async function deleteEvidence(evidenceId: string): Promise<void> {
	await getDatabase().evidence.delete(evidenceId);
}

export async function listEvidence(): Promise<Evidence[]> {
	return getDatabase().evidence.orderBy("createdAt").toArray();
}
