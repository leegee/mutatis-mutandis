import type { SelectionController } from "./SelectionController";

type ModifierKey = "shiftKey" | "ctrlKey" | "altKey" | "metaKey";

export class CanvasDragPlugin<T extends { event_id: string }> {
    private start: { x: number; y: number } | null = null;
    private modifierKey = 'ctrlKey' as ModifierKey;


    constructor(
        private canvas: HTMLCanvasElement,
        private deck: any,
        private controller: SelectionController<T>
    ) {
        canvas.addEventListener("mousedown", this.onDown);
        canvas.addEventListener("mouseup", this.onUp);
    }

    private getXY(e: MouseEvent) {
        const r = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - r.left,
            y: e.clientY - r.top,
        };
    }

    private onDown = (e: MouseEvent) => {
        // only care about ctrl drag
        if (!e[this.modifierKey]) return;
        this.start = this.getXY(e);
        this.controller.setDragStart?.(this.start);
        window.addEventListener("mousemove", this.onMove);
    };

    private onUp = (e: MouseEvent) => {
        if (!this.start) return;

        window.removeEventListener("mousemove", this.onMove);
        this.controller.setDragPreview?.(null);

        // must still be shift at release
        if (!e[this.modifierKey]) {
            this.start = null;
            return;
        }

        const end = this.getXY(e);

        const rect = {
            x: Math.min(this.start.x, end.x),
            y: Math.min(this.start.y, end.y),
            width: Math.abs(this.start.x - end.x),
            height: Math.abs(this.start.y - end.y),
        };

        this.controller.dispatch({
            type: "rect",
            payload: { rect, deck: this.deck },
        });

        this.start = null;
    };

    private onMove = (e: MouseEvent) => {
        if (!this.start || !e[this.modifierKey]) return;

        const current = this.getXY(e);

        this.controller.setDragPreview?.({
            x: Math.min(this.start.x, current.x),
            y: Math.min(this.start.y, current.y),
            width: Math.abs(this.start.x - current.x),
            height: Math.abs(this.start.y - current.y),
        });
    };

    destroy() {
        this.canvas.removeEventListener("mousedown", this.onDown);
        this.canvas.removeEventListener("mouseup", this.onUp);
    }
}
