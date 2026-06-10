import type { PointData } from "../types";
import type { Id, ScreenRect, SelectionEvent, SelectionOptions, SelectionPlugin } from "./types";

export class SelectionController<T extends { event_id: Id }> {
    private selected = new Set<Id>();
    private plugins: SelectionPlugin[] = [];
    private onChange?: (selected: Set<Id>) => void;

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

    // Event entry point
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
        }
    }

    // Core logic

    private handleBgClick() {
        this.clear();
    }

    private handleClick(obj: T) {
        const id = obj.event_id;
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
        const hits = deck.pickObjects(rect) as Array<{ object?: T }>;

        const additive =
            this.options.mode === "additive" && this.multiKeyDown;

        if (!additive) {
            this.clear();
        }

        for (const hit of hits) {
            const id = hit.object?.event_id;
            if (id != null) {
                this.selected.add(id);
            }
        }

        this.emit();
    }


    // State
    getSelected(): ReadonlySet<Id> {
        return this.selected;
    }

    setChangeHandler(fn: (s: Set<Id>) => void) {
        this.onChange = fn;
    }

    clear() {
        this.selected.clear();
        this.emit();
    }

    setDragPreview?: (rect: ScreenRect | null) => void;

    setDragStart?: (p: { x: number; y: number }) => void;

    // Internals
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
