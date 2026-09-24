import type { LineageNode } from "~/types/lineage";

export type TooltipState = {
	node: LineageNode;
	x: number;
	y: number;
};

export type ViewportRatio = {
	// Both 0..1, expressed as a fraction of the detail view's full
	// scrollable width -- not pixels, so they stay meaningful regardless
	// of container size or how many years are in the timeline.
	startRatio: number;
	endRatio: number;
};

export type ScrollState = {
	scrollLeft: number;
	scrollWidth: number;
	clientWidth: number;
};
