import type { PointData } from "../types";

export type Id = PointData["event_id"];

export interface ScreenRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface SelectionEvent<T> {
  type: "click" | "rect" | "background-click" | "null-select";
  payload: T;
}

export type SelectionMode = "replace" | "additive";

export interface SelectionOptions {
  mode?: SelectionMode;
  multiKey?: "Shift" | "Ctrl" | "Alt";
}

export interface SelectionPlugin {
  destroy(): void;
}
