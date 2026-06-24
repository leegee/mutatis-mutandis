// ScatterPlot/Plot.tsx
// Full-screen WebGL scatter plot for EEBO event data.
// Pure render component: all state lives in the parent.

import { createEffect, createMemo, createSignal, onCleanup, onMount } from "solid-js";
import { Deck, OrthographicView } from "@deck.gl/core";
import { ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import type { OrthographicViewState, PickingInfo } from "@deck.gl/core";
import { COORDINATE_SYSTEM } from "@deck.gl/core";

import "./style.css";
import type { BfsDataset, ConceptDataset, LabelPoint, LabelDataset, PointData, ViewBounds } from "./types";
import { CanvasDragPlugin } from "./SelectionPlugin/CanvasDragPlugin";
import { DeckClickPlugin } from "./SelectionPlugin/DeckClickPlugin";
import { SelectionController } from "./SelectionPlugin/SelectionController";
import type { Id, ScreenRect } from "./SelectionPlugin/types";
import { buildColorMap } from "../../lib/colour";
import type { ProjectionModeType } from "../../state/controls.store";

interface PlotProps {
  // Data
  datasets: ConceptDataset[];
  bfsDataset?: BfsDataset;
  labelDataset?: LabelDataset;

  // Controlled display state — parent owns these
  projectionMode: ProjectionModeType;
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
  minZoom: 8,
  maxZoom: 20,
};


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


  const labelLayer = props.labelDataset?.labels.length
    ? new TextLayer<LabelPoint>({
      id: "labels",
      data: props.labelDataset.labels,

      getPosition: d =>
        props.projectionMode === "global"
          ? [d.gnx ?? d.nx, d.gny ?? d.ny, 0]
          : [d.nx, d.ny, 0],

      getText: d => d.text,
      getSize: 16,
      sizeUnits: "pixels",
      getColor: [222, 222, 222, 222],
      getTextAnchor: "middle",
      getAlignmentBaseline: "center",
    })
    : null;


  // One ScatterplotLayer per concept dataset so toggling visibility
  // per-concept is trivially available to the parent later.
  const layers = createMemo(() => {
    const proj = props.projectionMode;
    const colorFn = getColor();
    const radius = props.pointRadius ?? 4;
    const opacity = props.opacity ?? 0.85;
    const bfsOpacity = props.bfsOpacity ?? 100;

    // console.debug("[Plot] bfsDataset points:", props.bfsDataset?.points?.length);

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

    const bfsLayer = props.bfsDataset
      && bfsOpacity
      && props.bfsDataset?.points?.length
      && props.projectionMode === "global"
      && new ScatterplotLayer<PointData>(
        {
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
        }
      );

    return [
      ...(bfsLayer ? [bfsLayer] : []),
      ...neighbourLayers,
      ...conceptLayers,
      ...(labelLayer ? [labelLayer] : []),
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
      const pick = await deck?.pickObjects({
        x: e.offsetX,
        y: e.offsetY,
      });

      const cleanPick = pick
        ?.filter((p) => !p.sourceLayer?.id.startsWith("bfs-"))
        .map((p) => p.object)
        .filter((o): o is PointData => !!o?.event_id) ?? [];

      if (cleanPick.length) {
        controller?.dispatch({ type: "click", payload: cleanPick });
      } else {
        controller?.dispatch({ type: "null-select", payload: null });
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
            border: "1px solid rgba(120,160,255,0.8)",
            "border-radius": 0,
            "background-color": "rgba(120,160,255,0.15)",
            "pointer-events": "none",
          }}
        />
      )}

    </article>
  );
}