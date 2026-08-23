export type InputField = "source" | "type" | "target";

export interface ParsedRow {
	source: string;
	type: string;
	target: string;
	parseError?: string;
}

export interface ImportRow extends ParsedRow {
	sourceExists: boolean;
	targetExists: boolean;
	relationTypeValid: boolean;
	relationExists: boolean;
}
