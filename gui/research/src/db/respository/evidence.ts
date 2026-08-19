import type { Evidence } from "~/domain/project";
import { getDatabase } from "../database";

export async function listEvidence(): Promise<Evidence[]> {
	return getDatabase().evidence.orderBy("createdAt").toArray();
}

export async function createEvidence(evidence: Evidence): Promise<Evidence> {
	await getDatabase().evidence.add(evidence);

	return evidence;
}
