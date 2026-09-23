// src/server/tier3/umap.ts

import type { UmapCluster, UmapDataset, UmapPoint } from "~/components/ScatterPlot/types";
import type { YearMode } from "~/types/controls";
import { get_connection } from "../db";

interface PointRow {
	event_id: number;
	token: string;
	pub_year: number | null;
	nx: number;
	ny: number;
	cluster_id: number | null;
}

interface ClusterRow {
	cluster_id: number;
	centroid_nx: number;
	centroid_ny: number;
	description: string | null;
	point_count: number;
}

export async function loadUmapData(
	concept: string,
	yearMode: YearMode,
	fromYear: number,
	toYear: number,
): Promise<UmapDataset> {
	if (!concept) {
		return {
			concept,
			points: [],
			clusters: [],
		};
	}

	const db = get_connection();

	let pointRows: PointRow[];

	if (yearMode === "single") {
		pointRows = await db<PointRow[]>`
		SELECT
			eg.event_id,
			e.token,
			e.pub_year,
			eg.nx,
			eg.ny,
			eg.cluster_id
		FROM tier3.event_geometry eg
		JOIN events e ON e.event_id = eg.event_id
		WHERE eg.concept = ${concept}
		  AND e.pub_year = ${fromYear}
		ORDER BY eg.event_id
	`;
	} else {
		pointRows = await db<PointRow[]>`
		SELECT
			eg.event_id,
			e.token,
			e.pub_year,
			eg.nx,
			eg.ny,
			eg.cluster_id
		FROM tier3.event_geometry eg
		JOIN events e ON e.event_id = eg.event_id
		WHERE eg.concept = ${concept}
		  AND e.pub_year BETWEEN ${fromYear} AND ${toYear}
		ORDER BY eg.event_id
	`;
	}

	const clusterRows = await db<ClusterRow[]>`
		SELECT
			cluster_id,
			centroid_nx,
			centroid_ny,
			description,
			point_count
		FROM tier3.concept_cluster_info
		WHERE concept = ${concept}
		ORDER BY cluster_id
	`;

	const points: UmapPoint[] = pointRows.map((row) => ({
		eventId: String(row.event_id),
		token: row.token,
		pubYear: row.pub_year === null ? null : Number(row.pub_year),
		x: Number(row.nx),
		y: Number(row.ny),
		clusterId: row.cluster_id === null ? null : Number(row.cluster_id),
	}));

	const clusters: UmapCluster[] = clusterRows.map((row) => ({
		clusterId: Number(row.cluster_id),
		x: Number(row.centroid_nx),
		y: Number(row.centroid_ny),
		label: null,
		pointCount: Number(row.point_count),
		description: row.description,
	}));

	return {
		concept,
		points,
		clusters,
	};
}
