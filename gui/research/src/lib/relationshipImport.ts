// lib/relationshipImport.ts

import type { RelationType } from "~/domain/relation";

export interface ParsedRelationship {
	source: string;
	type: string;
	target: string;
}

export interface RelationshipImportRow extends ParsedRelationship {
	sourceExists: boolean;
	targetExists: boolean;
	relationTypeValid: boolean;
	relationExists: boolean;
	parseError?: string;
}

const RELATIONSHIP_PATTERN =
	/^\s*(.*?)\s*(?:\|+|--+>|->|>)\s*(.*?)\s*(?:\|+|--+>|->|>)\s*(.*?)\s*$/;

export function parseRelationshipLine(
	line: string,
): ParsedRelationship | null {
	const match = line.match(RELATIONSHIP_PATTERN);
	if (!match) return null;
	const [, source, type, target] = match;

	if (!source?.trim() || !type?.trim() || !target?.trim()) return null;

	return {
		source: source.trim(),
		type: type.trim(),
		target: target.trim(),
	};
}

export function parseRelationships(text: string): Array<
	ParsedRelationship & { line: number; error?: string }
> {
	return text
		.split(/\r?\n/)
		.map((line, index) => ({
			line: index + 1,
			text: line,
		}))
		.filter(({ text }) => text.trim().length > 0)
		.map(({ line, text }) => {
			const parsed = parseRelationshipLine(text);

			if (!parsed) {
				return {
					line,
					source: "",
					type: "",
					target: "",
					error: `Could not parse line ${line}`,
				};
			}

			return {
				line,
				...parsed,
			};
		});
}

export function isRelationType(
	value: string,
	relationTypes: readonly RelationType[],
): value is RelationType {
	return relationTypes.includes(value as RelationType);
}
