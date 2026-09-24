// gui\corpus-gui\src\server\tier3\lineage.ts

import { query } from "@solidjs/router";
import type { ContextProfileEntry, LineageData, LineageEvent, LineageLink, LineageNode } from "~/types/lineage";
import { get_connection } from "../db";

const DRIFT_THRESHOLD = 0.75;
const CONFIDENCE_THRESHOLD = 0.95;
const EVENT_SAMPLE_SIZE = 8;
const CONTEXT_PROFILE_SIZE = 10;

interface ClusterRow {
	concept: string;
	pub_year: number;
	cluster_id: number;
	point_count: number;
	centroid_nx: number | null;
	centroid_ny: number | null;
	centroid_gnx: number | null;
	centroid_gny: number | null;
	centroid_vector: Uint8Array | null;
}

interface TemporalEdgeRow {
	source_year: number;
	source_cluster: number;
	target_year: number;
	target_cluster: number;
	similarity: number;
	confidence: number | null;
	edge_type: string;
}

interface EventRow {
	event_id: string;
	doc_id: string;
	token_idx: number;
	token: string;
	pub_year: number | null;
}

interface ContextRow {
	token: string;
	count: number;
}

function cosineSimilarity(a: Uint8Array, b: Uint8Array): number {
	const av = new Float32Array(a.buffer, a.byteOffset, a.byteLength / Float32Array.BYTES_PER_ELEMENT);
	const bv = new Float32Array(b.buffer, b.byteOffset, b.byteLength / Float32Array.BYTES_PER_ELEMENT);

	if (av.length !== bv.length || av.length === 0) {
		return 0;
	}

	let dot = 0;
	let normA = 0;
	let normB = 0;

	for (let i = 0; i < av.length; i++) {
		dot += av[i] * bv[i];
		normA += av[i] * av[i];
		normB += bv[i] * bv[i];
	}

	const denominator = Math.sqrt(normA) * Math.sqrt(normB);

	return denominator === 0 ? 0 : dot / denominator;
}

function nodeId(year: number, cluster: number): string {
	return `${year}:${cluster}`;
}

async function loadEvents(
	db: ReturnType<typeof get_connection>,
	concept: string,
	year: number,
	cluster: number,
): Promise<LineageEvent[]> {
	const rows = await db<EventRow[]>`
		SELECT
			e.event_id,
			e.doc_id,
			e.token_idx,
			e.token,
			e.pub_year
		FROM tier3.concept_year_event_cluster c
		JOIN events e
			ON e.event_id = c.event_id
		WHERE c.concept = ${concept}
		  AND c.pub_year = ${year}
		  AND c.cluster_id = ${cluster}
		ORDER BY e.event_id
	`;

	if (rows.length <= EVENT_SAMPLE_SIZE) {
		return rows.map((row) => ({
			event_id: row.event_id,
			doc_id: row.doc_id,
			token_idx: Number(row.token_idx),
			token: row.token,
			pub_year: row.pub_year === null ? null : Number(row.pub_year),
		}));
	}

	// Evenly sample the chronological event list so a dense cluster does
	// not make the response disproportionately large while preserving
	// deterministic output for the same database state.
	const indices = new Set<number>();

	for (let i = 0; i < EVENT_SAMPLE_SIZE; i++) {
		indices.add(Math.round((i * (rows.length - 1)) / (EVENT_SAMPLE_SIZE - 1)));
	}

	return [...indices]
		.sort((a, b) => a - b)
		.map((index) => {
			const row = rows[index];

			return {
				event_id: row.event_id,
				doc_id: row.doc_id,
				token_idx: Number(row.token_idx),
				token: row.token,
				pub_year: row.pub_year === null ? null : Number(row.pub_year),
			};
		});
}

async function loadContextProfile(
	db: ReturnType<typeof get_connection>,
	concept: string,
	year: number,
	cluster: number,
): Promise<ContextProfileEntry[]> {
	const rows = await db<ContextRow[]>`
		SELECT
			LOWER(e.token) AS token,
			COUNT(*) AS count
		FROM tier3.concept_year_event_cluster c
		JOIN events e
			ON e.event_id = c.event_id
		WHERE c.concept = ${concept}
		  AND c.pub_year = ${year}
		  AND c.cluster_id = ${cluster}
		  AND e.token <> ''
		GROUP BY LOWER(e.token)
		ORDER BY count DESC, token ASC
		LIMIT ${CONTEXT_PROFILE_SIZE}
	`;

	return rows.map((row) => ({
		token: row.token,
		count: Number(row.count),
	}));
}

export const getLineage = query(async (concept: string): Promise<LineageData> => {
	"use server";

	const started = performance.now();
	const db = get_connection();

	const clusters = await db<ClusterRow[]>`
			SELECT
				concept,
				pub_year,
				cluster_id,
				point_count,
				centroid_nx,
				centroid_ny,
				centroid_gnx,
				centroid_gny,
				centroid_vector
			FROM tier3.concept_year_cluster_info
			WHERE concept = ${concept}
			  AND cluster_id >= 0
			ORDER BY pub_year, cluster_id
		`;

	const temporalEdges = await db<TemporalEdgeRow[]>`
			SELECT
				source_year,
				source_cluster,
				target_year,
				target_cluster,
				similarity,
				confidence,
				edge_type
			FROM tier3.temporal_cluster_edges
			WHERE concept = ${concept}
			ORDER BY
				source_year,
				source_cluster,
				target_year,
				target_cluster
		`;

	const clusterById = new Map<string, ClusterRow>();

	for (const cluster of clusters) {
		clusterById.set(nodeId(cluster.pub_year, cluster.cluster_id), cluster);
	}

	// Keep only edges whose endpoints survived the cluster filter.
	// Tier 3.1 may contain references to noise clusters, but those are
	// deliberately absent from the lineage graph.
	const links: LineageLink[] = [];

	for (const edge of temporalEdges) {
		const source = nodeId(edge.source_year, edge.source_cluster);
		const target = nodeId(edge.target_year, edge.target_cluster);

		if (!clusterById.has(source) || !clusterById.has(target)) {
			continue;
		}

		links.push({
			source,
			target,
			similarity: Number(edge.similarity),
			confidence: edge.confidence === null ? 0 : Number(edge.confidence),
			type: edge.edge_type,
		});
	}

	const incoming = new Map<string, LineageLink[]>();
	const outgoing = new Map<string, LineageLink[]>();

	for (const link of links) {
		const incomingLinks = incoming.get(link.target) ?? [];
		incomingLinks.push(link);
		incoming.set(link.target, incomingLinks);

		const outgoingLinks = outgoing.get(link.source) ?? [];
		outgoingLinks.push(link);
		outgoing.set(link.source, outgoingLinks);
	}

	const lineageByNode = new Map<string, number>();
	const mergedFromByNode = new Map<string, number[]>();
	const persistenceByNode = new Map<string, number>();
	const lineageAnchor = new Map<number, ClusterRow>();
	const lineageFounder = new Map<number, string>();
	let nextLineage = 0;

	const sortedClusters = [...clusters].sort((a, b) => a.pub_year - b.pub_year || a.cluster_id - b.cluster_id);

	for (const cluster of sortedClusters) {
		const id = nodeId(cluster.pub_year, cluster.cluster_id);

		const parents = incoming.get(id) ?? [];

		if (parents.length === 0) {
			lineageByNode.set(id, nextLineage);
			lineageAnchor.set(nextLineage, cluster);
			lineageFounder.set(nextLineage, id);
			persistenceByNode.set(id, 1);
			nextLineage++;
			continue;
		}

		const ranked = [...parents].sort((a, b) => b.confidence - a.confidence);

		const parent = ranked[0];
		const parentLineage = lineageByNode.get(parent.source);

		if (parentLineage === undefined) {
			throw new Error(`Lineage parent has not been processed: ${parent.source}`);
		}

		const anchor = lineageAnchor.get(parentLineage);

		let persistence = parent.confidence;

		if (cluster.centroid_vector && anchor?.centroid_vector) {
			persistence = cosineSimilarity(cluster.centroid_vector, anchor.centroid_vector);
		}

		const continuation = parent.confidence >= CONFIDENCE_THRESHOLD && persistence >= DRIFT_THRESHOLD;

		if (!continuation) {
			lineageByNode.set(id, nextLineage);
			lineageAnchor.set(nextLineage, cluster);
			lineageFounder.set(nextLineage, id);
			persistenceByNode.set(id, 1);
			nextLineage++;
		} else {
			lineageByNode.set(id, parentLineage);
			persistenceByNode.set(id, persistence);
		}

		const otherLineages = [
			...new Set(
				ranked
					.slice(1)
					.map((edge) => lineageByNode.get(edge.source))
					.filter((lid): lid is number => lid !== undefined && lid !== lineageByNode.get(id)),
			),
		].sort((a, b) => a - b);

		if (otherLineages.length > 0) {
			mergedFromByNode.set(id, otherLineages);
		}
	}

	const lineageMinPersistence = new Map<number, number>();

	for (const [id, lineage] of lineageByNode) {
		const score = persistenceByNode.get(id) ?? 1;

		const current = lineageMinPersistence.get(lineage);

		lineageMinPersistence.set(lineage, current === undefined ? score : Math.min(current, score));
	}

	const lineageStable = new Map<number, boolean>();

	for (const [lineage, persistence] of lineageMinPersistence) {
		lineageStable.set(lineage, persistence >= DRIFT_THRESHOLD);
	}

	const nodes: LineageNode[] = [];

	for (const cluster of sortedClusters) {
		const id = nodeId(cluster.pub_year, cluster.cluster_id);

		const lineage = lineageByNode.get(id);

		if (lineage === undefined) {
			throw new Error(`No lineage assigned to ${id}`);
		}

		const [eventSample, contextProfile] = await Promise.all([
			loadEvents(db, concept, cluster.pub_year, cluster.cluster_id),
			loadContextProfile(db, concept, cluster.pub_year, cluster.cluster_id),
		]);

		nodes.push({
			id,
			year: cluster.pub_year,
			cluster: cluster.cluster_id,
			size: cluster.point_count,
			lineage,
			merged_from: mergedFromByNode.get(id) ?? [],
			persistence_score: persistenceByNode.get(id) ?? 1,
			lineage_stable: lineageStable.get(lineage) ?? false,
			local: {
				x: cluster.centroid_nx ?? 0,
				y: cluster.centroid_ny ?? 0,
			},
			global: {
				x: cluster.centroid_gnx ?? 0,
				y: cluster.centroid_gny ?? 0,
			},
			event_sample: eventSample,
			context_profile: contextProfile,
		});
	}

	const births = nodes.filter((node) => (incoming.get(node.id) ?? []).length === 0).map((node) => node.id);

	const deaths = nodes.filter((node) => (outgoing.get(node.id) ?? []).length === 0).map((node) => node.id);

	const branching = nodes
		.map((node) => [node.id, (outgoing.get(node.id) ?? []).length] as [string, number])
		.filter(([, count]) => count > 1);

	const merging = nodes
		.map((node) => [node.id, (incoming.get(node.id) ?? []).length] as [string, number])
		.filter(([, count]) => count > 1);

	const unstable = [...lineageStable.entries()].filter(([, stable]) => !stable).map(([lineage]) => lineage);

	const lineages = [...lineageMinPersistence.entries()]
		.sort(([a], [b]) => a - b)
		.map(([lineage, minPersistence]) => ({
			lineage,
			min_persistence: minPersistence,
			stable: lineageStable.get(lineage) ?? false,
		}));

	const elapsed = (performance.now() - started) / 1000;

	return {
		generated: "tier4_0_lineage_graph",
		concept,
		nodes,
		links,
		lineages,
		drift_threshold: DRIFT_THRESHOLD,
		confidence_threshold: CONFIDENCE_THRESHOLD,
		summary: {
			nodes: nodes.length,
			edges: links.length,
			lineages: lineages.length,
			stable_lineages: lineages.filter((lineage) => lineage.stable).length,
			births: births.length,
			deaths: deaths.length,
			branching: branching.length,
			merging: merging.length,
			unstable_lineages: unstable.length,
		},
		events: {
			births,
			deaths,
			branching,
			merging,
			unstable,
		},
		elapsed_seconds: Number(elapsed.toFixed(3)),
	};
}, "lineage");
