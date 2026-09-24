// gui\corpus-gui\src\types\lineage.ts

export interface LineageEvent {
	event_id: string;
	doc_id: string;
	token_idx: number;
	token: string;
	pub_year: number | null;
	context: string;
}

export interface ContextProfileEntry {
	token: string;
	count: number;
	score: number;
}

export interface LineageNode {
	id: string;
	year: number;
	cluster: number;
	size: number;
	lineage: number;
	merged_from: number[];
	persistence_score: number;
	lineage_stable: boolean;
	local: {
		x: number;
		y: number;
	};
	global: {
		x: number;
		y: number;
	};
	event_sample: LineageEvent[];
	context_profile: ContextProfileEntry[];
}

export interface LineageLink {
	source: string;
	target: string;
	similarity: number;
	confidence: number;
	type: string;
}

export interface LineageSummary {
	lineage: number;
	min_persistence: number;
	stable: boolean;
}

export interface LineageAnalysis {
	births: string[];
	deaths: string[];
	branching: [string, number][];
	merging: [string, number][];
	unstable: number[];
}

export interface LineageData {
	generated: string;
	concept: string;
	nodes: LineageNode[];
	links: LineageLink[];
	lineages: LineageSummary[];
	drift_threshold: number;
	confidence_threshold: number;
	summary: {
		nodes: number;
		edges: number;
		lineages: number;
		stable_lineages: number;
		births: number;
		deaths: number;
		branching: number;
		merging: number;
		unstable_lineages: number;
	};
	events: LineageAnalysis;
	elapsed_seconds: number;
}
