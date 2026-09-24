export interface Neighbour {
	eventId: string;
	docId: string;
	tokenIdx: string;
	token: string;
	pubYear: number;
	// count?: number;
	// max_score?: number;
	score?: number;
}

export interface NeighbourhoodEvent {
	eventId: string;
	docId: string;
	token: string;
	tokenIdx: number;
	pubYear: number;
	neighbours: Neighbour[];
}

export interface NeighbourhoodData {
	events: NeighbourhoodEvent[];
	yearBounds: [number, number];
}
