// ScatterPlot/Plot.tsx
// Full-screen WebGL scatter plot for EEBO event data.
// Pure render component: all state lives in the parent.

import { createEffect, createMemo, createSignal, onCleanup, onMount } from "solid-js";
import { Deck, OrthographicView } from "@deck.gl/core";
import { ScatterplotLayer } from "@deck.gl/layers";
import type { OrthographicViewState, PickingInfo } from "@deck.gl/core";
import { COORDINATE_SYSTEM } from "@deck.gl/core";

import "./style.css";
import type { BfsDataset, ConceptDataset, PointData, ViewBounds } from "./types";
import { CanvasDragPlugin } from "./SelectionPlugin/CanvasDragPlugin";
import { DeckClickPlugin } from "./SelectionPlugin/DeckClickPlugin";
import { SelectionController } from "./SelectionPlugin/SelectionController";
import type { Id, ScreenRect } from "./SelectionPlugin/types";

interface PlotProps {
  // Data
  datasets: ConceptDataset[];
  bfsDataset?: BfsDataset;

  // Controlled display state — parent owns these
  projection: "local" | "global";
  colorBy: string;
  colorByFields: string[];
  pointRadius?: number;
  opacity?: number;
  bfsOpacity?: number
  neighbourOpacity?: number;
  selected?: Set<Id> | null;

  // Events
  onPointHover?: (point: PointData | null, screenXY: [number, number] | null) => void;
  onPointRightClick?: (point: PointData) => void;
  onBoundsChange?: (bounds: ViewBounds) => void;
  onSelectionChange?: (ids: string[] | null) => void;
}


const GREY: [number, number, number, number] = [120, 120, 130, 140];
const NEIGHBOURS: [number, number, number] = [120, 120, 170];
const getBfsPosition = (p: PointData) => [p.gnx, p.gny, 0] as [number, number, number];
const INITIAL_VIEW_STATE: OrthographicViewState = {
  target: [0.5, 0.5, 0],
  zoom: 10,
  minZoom: -2,
  maxZoom: 20,
};

// Deterministic hash so the same value always maps to the same colour,
// regardless of dataset load order or concept switching.
function hashString(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

// Generates a perceptually spread palette using golden-ratio hue stepping.
// Returns [r, g, b, a] arrays suitable for deck.gl.
function buildColorMap(
  values: string[],
  topN = 24
): Map<string, [number, number, number, number]> {
  const counts = new Map<string, number>();
  for (const v of values) counts.set(v, (counts.get(v) ?? 0) + 1);

  const sorted = [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([v]) => v);

  const map = new Map<string, [number, number, number, number]>();
  const golden = 0.6180339887;

  sorted.forEach((v, i) => {
    const hue = ((hashString(v) / 0xffffffff + i * golden) % 1) * 360;
    map.set(v, hslToRgb(hue, 0.72, 0.62));
  });

  return map;
}

function hslToRgb(
  h: number,
  s: number,
  l: number
): [number, number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  if (h < 60) { r = c; g = x; }
  else if (h < 120) { r = x; g = c; }
  else if (h < 180) { g = c; b = x; }
  else if (h < 240) { g = x; b = c; }
  else if (h < 300) { r = x; b = c; }
  else { r = c; b = x; }
  return [
    Math.round((r + m) * 255),
    Math.round((g + m) * 255),
    Math.round((b + m) * 255),
    220,
  ];
}

function getPosition(
  p: PointData,
  projection: "local" | "global"
): [number, number, number] {
  return projection === "global"
    ? [p.gnx, p.gny, 0]
    : [p.nx, p.ny, 0];
}

export default function Plot(props: PlotProps) {
  let canvas!: HTMLCanvasElement;
  let deck: Deck<OrthographicView> | null = null;
  let controller: SelectionController<PointData> | undefined;

  const [_viewState, setViewState] = createSignal<OrthographicViewState>(INITIAL_VIEW_STATE);
  const [dragRect, setDragRect] = createSignal<ScreenRect | null>(null);

  // Flatten all points across all datasets for colour map derivation.
  // const allPoints = createMemo<PointData[]>(() =>
  //   props.datasets.flatMap((d) => d.points)
  // );
  const allPoints = createMemo<PointData[]>(() =>
    props.datasets.flatMap((d) =>
      (d.points ?? []).filter(
        (p): p is PointData => p != null
      )
    )
  );

  // Build colour map whenever the field or data changes.
  const colorMap = createMemo(() => {
    const field = props.colorBy;
    const values = allPoints().map((p) => String(p[field] ?? ""));
    return buildColorMap(values);
  });

  const selected = createMemo(() => props.selected ?? new Set<Id>());

  const getColor = createMemo(() => {
    const field = props.colorBy;
    const map = colorMap();
    const sel = selected();

    return (p: PointData, origin?: string): [number, number, number, number] => {
      const base =
        origin === "neighbours"
          ? [...NEIGHBOURS, props.neighbourOpacity ?? 200]
          : map.get(String(p[field] ?? "")) ?? GREY;

      // selection emphasis
      if (sel.size > 0) {
        if (sel.has(p.event_id)) {
          return [
            Math.min(base[0] * 1.25, 255),
            Math.min(base[1] * 1.25, 255),
            Math.min(base[2] * 1.25, 255),
            255
          ];
        } else {
          return [
            Math.min(base[0] * .5, base[0]),
            Math.min(base[1] * .5, base[1]),
            Math.min(base[2] * .5, base[2]),
            Math.min(base[3]),
          ]
        }
      }
      return base as [number, number, number, number];
    }
  });


  // One ScatterplotLayer per concept dataset so toggling visibility
  // per-concept is trivially available to the parent later.
  const layers = createMemo(() => {
    const proj = props.projection;
    const colorFn = getColor();
    const radius = props.pointRadius ?? 4;
    const opacity = props.opacity ?? 0.85;
    const bfsOpacity = props.bfsOpacity ?? 100;

    console.log("[Plot] bfsDataset points:", props.bfsDataset?.points?.length);

    const [neighbourLayers, conceptLayers] = props.datasets.reduce(
      ([n, c], dataset) => {
        const layer = new ScatterplotLayer<PointData>({
          id: `${ dataset.origin ?? "concept" }-${ dataset.concept }`,
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          data: dataset.points,
          getPosition: (p) => getPosition(p, proj),
          getFillColor: (p) => colorFn(p, dataset.origin),
          getRadius: radius,
          radiusUnits: "pixels",
          opacity,
          pickable: true,
          autoHighlight: true,
          highlightColor: [255, 255, 255, 80],
          transitions: {
            getPosition: { duration: 650, easing: (t: number) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t },
            getFillColor: { duration: 300 },
          },
          updateTriggers: {
            getPosition: [proj],
            getFillColor: [props.neighbourOpacity, props.colorBy, dataset.concept, dataset.origin, selected()],
          },
          onHover: (info: PickingInfo<PointData>) => props.onPointHover?.(info.object ?? null, info.object ? [info.x, info.y] : null),
        });

        return dataset.origin === "neighbours"
          ? [[...n, layer], c]
          : [n, [...c, layer]];
      },
      [[], []] as [ScatterplotLayer<PointData>[], ScatterplotLayer<PointData>[]]
    );

    const bfsLayer = props.bfsDataset && props.projection === "global" && new ScatterplotLayer<PointData>({
      id: "bfs-global",
      coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
      data: props.bfsDataset.points,
      getPosition: getBfsPosition,
      getFillColor: () => [150, 150, 150, bfsOpacity],
      updateTriggers: {
        getFillColor: bfsOpacity,
      },
      getRadius: radius * 1.5,
      radiusUnits: "pixels",
      pickable: true,
      onHover: (info: PickingInfo<PointData>) => props.onPointHover?.(info.object ?? null, info.object ? [info.x, info.y] : null),
    });

    return [
      ...(bfsLayer ? [bfsLayer] : []),
      ...neighbourLayers,
      ...conceptLayers,
    ];
  });

  function fitZoom(): number {
    const size = Math.min(canvas.clientWidth, canvas.clientHeight) * 0.9;
    return Math.log2(size / 512);
  }


  // Emit onBoundsChange when the viewport moves.
  function handleViewStateChange({ viewState: vs }: { viewState: OrthographicViewState }) {
    setViewState(vs);

    if (props.onBoundsChange) {
      const zoom = (vs.zoom as number) ?? 1;
      const scale = Math.pow(2, zoom);
      const halfW = (window.innerWidth / 2) / (512 * scale);
      const halfH = (window.innerHeight / 2) / (512 * scale);
      const [cx, cy] = vs.target as [number, number, number];
      props.onBoundsChange({
        minX: cx - halfW,
        maxX: cx + halfW,
        minY: cy - halfH,
        maxY: cy + halfH,
        zoom,
      });
    }
  }

  onMount(() => {
    deck = new Deck<OrthographicView>({
      canvas,
      views: new OrthographicView({ id: "ortho", controller: true }),
      initialViewState: INITIAL_VIEW_STATE,
      useDevicePixels: true,
      layers: [],                        // start empty — effect below sets immediately
      onViewStateChange: handleViewStateChange,
      style: { width: "100%", height: "100%" },
    });

    controller = new SelectionController<PointData>({
      mode: "additive",
      multiKey: "Shift",
    });

    controller.setChangeHandler((set) => props.onSelectionChange?.(set ? [...set] : null));
    controller.setDragPreview = (rect) => setDragRect(rect);
    controller
      .use(new DeckClickPlugin(deck, controller))
      .use(new CanvasDragPlugin(canvas, deck, controller));

    canvas.addEventListener("pointerup", async (e) => {
      const pick = await deck?.pickObjectsAsync({
        x: e.offsetX,
        y: e.offsetY,
      });

      if (pick) {
        controller?.dispatch({ type: "click", payload: pick });
      } else {
        controller?.dispatch({ type: "background-click", payload: null });
      }
    });

    fitZoom();
  });

  // Single source of truth for all layers, including BFS.
  createEffect(() => {
    deck?.setProps({ layers: layers() });
  });

  onCleanup(() => {
    deck?.finalize();
    deck = null;
  });

  return (
    <article id="UmapPlot">
      <canvas ref={canvas} />

      {dragRect() && (
        <div
          style={{
            position: "absolute",
            "z-index": 10,
            left: `${ dragRect()!.x }px`,
            top: `${ dragRect()!.y }px`,
            width: `${ dragRect()!.width }px`,
            height: `${ dragRect()!.height }px`,
            border: "2px solid rgba(120,160,255,1)",
            "background-color": "rgba(120,160,255,0.15)",
            "pointer-events": "none",
          }}
        />
      )}

    </article>
  );
}