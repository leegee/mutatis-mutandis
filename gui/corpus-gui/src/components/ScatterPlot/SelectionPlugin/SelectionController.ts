import type { ScreenRect, SelectionEvent, SelectionOptions, SelectionPlugin } from "./types";

export class SelectionController<T extends { eventId: string }> {
	private selected = new Set<string>();
	private plugins: SelectionPlugin[] = [];
	private onChange?: (selected: Set<string>) => void;

	private options: Required<SelectionOptions>;
	private multiKeyDown = false;

	constructor(options: SelectionOptions = {}) {
		this.options = {
			mode: options.mode ?? "additive",
			multiKey: options.multiKey ?? "Shift",
		};

		window.addEventListener("keydown", this.onKeyDown);
		window.addEventListener("keyup", this.onKeyUp);
	}

	use(plugin: SelectionPlugin) {
		this.plugins.push(plugin);
		return this;
	}

	dispatch(event: SelectionEvent<any>) {
		switch (event.type) {
			case "background-click":
				this.handleBgClick();
				break;

			case "click":
				this.handleClick(event.payload as T);
				break;

			case "rect":
				this.handleRect(event.payload.rect, event.payload.deck);
				break;

			case "null-select":
				this.handleNullSelect();
				break;
		}
	}

	private handleBgClick() {
		this.clear();
	}

	private handleNullSelect() {
		this.clear();
	}

	private handleClick(obj: T) {
		const id = obj.eventId;
		const additive = this.options.mode === "additive" && this.multiKeyDown;

		if (!additive) {
			this.clear();
		}

		if (additive && this.selected.has(id)) {
			this.selected.delete(id);
		} else {
			this.selected.add(id);
		}

		this.emit();
	}

	private handleRect(rect: ScreenRect, deck: any) {
		type Hit = { object?: T; layer?: { id: string } };
		const hits = deck.pickObjects(rect) as Array<Hit>;

		const additive = this.options.mode === "additive" && this.multiKeyDown;

		if (!additive) {
			this.clear();
		}

		for (const hit of hits) {
			if (hit.layer?.id.startsWith("bfs-")) continue;

			const id = hit.object?.eventId;

			if (id) {
				this.selected.add(id);
			}
		}

		this.emit();
	}

	getSelected(): ReadonlySet<string> {
		return this.selected;
	}

	setChangeHandler(fn: (selected: Set<string>) => void) {
		this.onChange = fn;
	}

	clear() {
		this.selected.clear();
		this.emit();
	}

	setDragPreview?: (rect: ScreenRect | null) => void;

	setDragStart?: (p: { x: number; y: number }) => void;

	private emit() {
		this.onChange?.(new Set(this.selected));
	}

	private onKeyDown = (e: KeyboardEvent) => {
		if (e.key === this.options.multiKey) {
			this.multiKeyDown = true;
		}
	};

	private onKeyUp = (e: KeyboardEvent) => {
		if (e.key === this.options.multiKey) {
			this.multiKeyDown = false;
		}
	};

	destroy() {
		window.removeEventListener("keydown", this.onKeyDown);
		window.removeEventListener("keyup", this.onKeyUp);

		for (const p of this.plugins) p.destroy();
		this.plugins = [];
	}
}
