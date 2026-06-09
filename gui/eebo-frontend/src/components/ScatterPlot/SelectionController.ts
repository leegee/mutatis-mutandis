/*
import { createSignal, onCleanup } from "solid-js";
import DeckGL from "@deck.gl/react";

type MyPoint = {
  id: string;
  position: [number, number];
};

const [selected, setSelected] = createSignal<Set<string>>(new Set());

let deckRef: any;

const controller = new SelectionController<MyPoint>(deckRef, {
  mode: "additive",
  multiKey: "Shift"
});

controller.setChangeHandler(setSelected);

onCleanup(() => controller.destroy());

function handlePointerUp(start: {x:number,y:number}, end: {x:number,y:number}) {
  controller.onDragEnd({
    x: Math.min(start.x, end.x),
    y: Math.min(start.y, end.y),
    width: Math.abs(start.x - end.x),
    height: Math.abs(start.y - end.y)
  });
}

<DeckGL
  ref={deckRef}
  layers={[scatterLayer]}
  onClick={controller.onClick}
/>

*/

/* IN LAYER:

const scatterLayer = new ScatterplotLayer({
  getFillColor: d => selected().has(d.id) ? [255, 80, 80]: [80, 140, 255],

  id: "points",
  data,
  pickable: true,
  getPosition: d => d.position,
  radiusMinPixels: 4
});
*/

type Id = string | number;

/** Object shape expected from deck.gl picking */
export interface PickedObject<T = any> {
  object?: T;
  layer?: unknown;
  index?: number;
  x?: number;
  y?: number;
}

/** Rectangle in screen space (pixels) */
export interface ScreenRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export type SelectionMode = "replace" | "additive";

export interface SelectionOptions {
  /** Default selection mode when no modifier is pressed */
  mode?: SelectionMode;

  /** Modifier key for multi-select */
  multiKey?: "Shift" | "Ctrl" | "Alt";
}

export class SelectionController<T extends { id: Id }> {
  private deck: any;

  private selected = new Set<Id>();
  private shiftDown = false;

  private options: Required<SelectionOptions>;

  private onChange?: (selected: Set<Id>) => void;

  constructor(deck: any, options: SelectionOptions = {}) {
    this.deck = deck;

    this.options = {
      mode: options.mode ?? "additive",
      multiKey: options.multiKey ?? "Shift"
    };

    window.addEventListener("keydown", this.onKeyDown);
    window.addEventListener("keyup", this.onKeyUp);
  }

  private onKeyDown = (e: KeyboardEvent) => {
    if (e.key === this.options.multiKey) {
      this.shiftDown = true;
    }
  };

  private onKeyUp = (e: KeyboardEvent) => {
    if (e.key === this.options.multiKey) {
      this.shiftDown = false;
    }
  };


  public onClick = (info: PickedObject<T>) => {
    const obj = info.object;
    if (!obj) return;

    const id = obj.id;

    if (this.shiftDown) {
      this.toggle(id);
    } else {
      this.selected.clear();
      this.selected.add(id);
    }

    this.emit();
  };


  public onDragEnd = (rect: ScreenRect) => {
    const hits = this.deck.pickObjects(rect) as Array<PickedObject<T>>;

    // If not multi-select, replace selection
    if (!this.shiftDown) {
      this.selected.clear();
    }

    for (const hit of hits) {
      const id = hit.object?.id;
      if (id !== undefined) {
        this.selected.add(id);
      }
    }

    this.emit();
  };


  private toggle(id: Id) {
    if (this.selected.has(id)) {
      this.selected.delete(id);
    } else {
      this.selected.add(id);
    }
  }

  private emit() {
    this.onChange?.(new Set(this.selected));
  }

  public setChangeHandler(fn: (selected: Set<Id>) => void) {
    this.onChange = fn;
  }

  public getSelected(): ReadonlySet<Id> {
    return this.selected;
  }

  public clear() {
    this.selected.clear();
    this.emit();
  }

  public destroy() {
    window.removeEventListener("keydown", this.onKeyDown);
    window.removeEventListener("keyup", this.onKeyUp);
  }
}

