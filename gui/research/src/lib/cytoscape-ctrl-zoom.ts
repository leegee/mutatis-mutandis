import type { Core } from "cytoscape";

export interface CtrlZoomOptions {
	duration?: number;
	minSize?: number;
}

export default function ctrlZoom(cytoscape: typeof import("cytoscape")) {
	cytoscape("core", "ctrlZoomBox", function (this: Core, options: CtrlZoomOptions = {}) {
		const duration = options.duration ?? 250;
		const minSize = options.minSize ?? 10;

		let start: { x: number; y: number } | undefined;

		this.on("boxstart", (event) => {
			const originalEvent = event.originalEvent as MouseEvent | undefined;

			if (!originalEvent?.ctrlKey) {
				start = undefined;
				return;
			}

			start = {
				x: event.renderedPosition.x,
				y: event.renderedPosition.y,
			};
		});

		this.on("boxend", (event) => {
			if (!start) return;

			const end = {
				x: event.renderedPosition.x,
				y: event.renderedPosition.y,
			};

			const left = Math.min(start.x, end.x);
			const right = Math.max(start.x, end.x);
			const top = Math.min(start.y, end.y);
			const bottom = Math.max(start.y, end.y);

			start = undefined;

			const width = right - left;
			const height = bottom - top;

			if (width < minSize || height < minSize) return;

			const currentZoom = this.zoom();
			const currentPan = this.pan();

			const modelLeft = (left - currentPan.x) / currentZoom;
			const modelRight = (right - currentPan.x) / currentZoom;
			const modelTop = (top - currentPan.y) / currentZoom;
			const modelBottom = (bottom - currentPan.y) / currentZoom;

			const modelWidth = modelRight - modelLeft;
			const modelHeight = modelBottom - modelTop;

			let newZoom = Math.min(this.width() / modelWidth, this.height() / modelHeight);

			newZoom = Math.max(this.minZoom(), Math.min(this.maxZoom(), newZoom));

			const centerX = (modelLeft + modelRight) / 2;
			const centerY = (modelTop + modelBottom) / 2;

			this.animate(
				{
					zoom: newZoom,
					pan: {
						x: this.width() / 2 - centerX * newZoom,
						y: this.height() / 2 - centerY * newZoom,
					},
				},
				{ duration },
			);
		});
	});
}

declare module "cytoscape" {
	interface Core {
		ctrlZoomBox(options?: CtrlZoomOptions): void;
	}
}
