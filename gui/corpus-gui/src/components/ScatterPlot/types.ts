export interface UmapPoint {
	eventId: string;
	token: string;
	pubYear: number | null;
	x: number;
	y: number;
	clusterId: number | null;
}

export interface UmapCluster {
	clusterId: number;
	x: number;
	y: number;
	label: string | null;
	pointCount: number;
	description: string | null;
}

export interface UmapDataset {
	concept: string;
	points: UmapPoint[];
	clusters: UmapCluster[];
}
