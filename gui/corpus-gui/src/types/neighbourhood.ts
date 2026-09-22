export interface Neighbour {
	eventId: number;
	docId: string;
	token: string;
	tokenIdx: number;
	pubYear: number | null;
	score: number;
}

export interface NeighbourhoodEvent {
	eventId: number;
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
