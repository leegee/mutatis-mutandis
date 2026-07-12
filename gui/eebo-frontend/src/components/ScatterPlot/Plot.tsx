// ScatterPlot/Plot.tsx
// Full-screen WebGL scatter plot for event data.
// Pure render component: all state lives in the parent.

import { createEffect, createMemo, createSignal, onCleanup, onMount } from "solid-js";
import { Deck, OrthographicView, LinearInterpolator } from "@deck.gl/core";
import { ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import type { OrthographicViewState, PickingInfo } from "@deck.gl/core";

import { GlowScatterplotLayer } from './GlowScatterplotLayer';

import "./style.css";
import type { BfsDataset, ConceptDatasetSqlite, LabelPoint, PointData, ViewBounds } from "./types";
import { CanvasDragPlugin } from "./SelectionPlugin/CanvasDragPlugin";
import { DeckClickPlugin } from "./SelectionPlugin/DeckClickPlugin";
import { SelectionController } from "./SelectionPlugin/SelectionController";
import type { Id, ScreenRect } from "./SelectionPlugin/types";
import { buildColorMap } from "../../lib/colour";
import { controls, type ProjectionModeType } from "../../state/controls.store";
import { labelsActions } from "../../state/labels.actions";

type RGB = [number, number, number];
type RGBA = [number, number, number, number];


const ZOOM_THRESHOLD = 7;
const DRAG_THRESHOLD_PX = 6;


const DEPTH_COLORS: Record<number, RGBA> = {
  0: [0, 0, 0, 0],
  1: [80, 160, 220, 250],
  2: [80, 200, 140, 250],
};

const GREY: RGBA = [120, 120, 130, 140];

const INITIAL_VIEW_STATE: OrthographicViewState = {
  target: [0.5, 0.5, 0],
  zoom: 10,
  minZoom: 8,
  maxZoom: 20,
};

const brighten = ([r, g, b]: RGB | RGBA): RGBA => [
  Math.min(r * 1.25, 255),
  Math.min(g * 1.25, 255),
  Math.min(b * 1.25, 255),
  255,
];

const dim = ([r, g, b, a]: RGBA): RGBA => [
  r * 0.5,
  g * 0.5,
  b * 0.5,
  a,
];

interface PlotProps {
  datasets: ConceptDatasetSqlite[];
  bfsDataset?: BfsDataset;

  projectionMode: ProjectionModeType;
  colorBy: string;
  colorByFields: string[];
  pointRadius?: number;
  bfsOpacity?: number;
  neighbourOpacity?: number;
  selectedEventIds?: Set<Id>;

  onPointRightClick?: (point: PointData) => void;
  onBoundsChange?: (bounds: ViewBounds) => void;
  onSelectionChange?: (points: PointData[] | null) => void;
  onPointHover?: (point: PointData | null, screenXY: [number, number] | null) => void;
  onLabelHover: (label: LabelPoint | null, screenXY: [number, number] | null) => void;
}

const getBfsPosition = (p: PointData) => [p.gnx, p.gny, 0] as [number, number, number];

const getPosition = (p: PointData | LabelPoint, projection: ProjectionModeType): RGB =>
  projection === "global" ? [p.gnx, p.gny, 0] : [p.nx, p.ny, 0];


export default function Plot(props: PlotProps) {
  let canvas!: HTMLCanvasElement;
  let deck: Deck<OrthographicView> | null = null;
  let controller: SelectionController<PointData> | undefined;

  let currentPoints: PointData[] = [];

  // Separate signal that only flips when the threshold actually crosses,
  // used solely to trigger a layer rebuild on that one frame.
  const [zoomTier, setZoomTier] = createSignal(0); // 0 = zoomed-out, 1 = zoomed-in
  let isZoomedIn = false;

  // Track drag state via pointer distance rather than a boolean flag.
  let pointerDownX = 0;
  let pointerDownY = 0;

  // Separate drag-preview signal for the selection rectangle overlay.
  let isDragging = false;
  const [dragRect, setDragRect] = createSignal<ScreenRect | null>(null);

  const allPoints = createMemo(() => {
    const start = performance.now();
    const result = props.datasets.flatMap(d => d.points || []);
    console.debug(`[Plot] allPoints memo computed ${ result.length } points in ${ (performance.now() - start).toFixed(1) }ms`);
    return result;
  });

  const selectedEventIds = createMemo(() => props.selectedEventIds ?? new Set<Id>());

  // Split so colorMap only rebuilds when the colour field values change,
  // not on every unrelated dataset mutation.
  const colorFieldValues = createMemo(() => {
    const field = props.colorBy;
    return allPoints().map(p => String(p[field as keyof PointData] ?? ""));
  });

  const colorMap = createMemo(() => buildColorMap(colorFieldValues()));

  const getColor = createMemo(() => {
    return (p: PointData, origin?: string): RGBA => {
      if (props.colorBy === "cluster_label" && p.cluster_id !== undefined && p.cluster_id !== -1) {
        const label = p.cluster_label || String(p.cluster_id);
        return colorMap().get(label) ?? GREY;
      }

      let base: RGBA;
      if (origin === "neighbours") {
        const thisNeighbourOpacity = props.neighbourOpacity ?? 200;
        const depth = p.depth ?? 1;
        const alpha = depth === 2
          ? Math.floor(thisNeighbourOpacity * 0.45)
          : thisNeighbourOpacity;
        base = [
          DEPTH_COLORS[depth][0],
          DEPTH_COLORS[depth][1],
          DEPTH_COLORS[depth][2],
          alpha,
        ];
      } else {
        const thisColourMap = colorMap();
        const thisColorBy = props.colorBy;
        base = thisColourMap.get(String(p[thisColorBy as keyof PointData] ?? "")) ?? GREY;
      }

      const thisSelectedEventIds = selectedEventIds();
      if (!thisSelectedEventIds.size) return base;
      if (thisSelectedEventIds.has(p.event_id)) return brighten(base);
      return dim(base);
    };
  });


  const labelLayer = createMemo(() => {
    const labels = labelsActions.getLabels(controls.concept);

    return new TextLayer<LabelPoint>({
      coordinateSystem: "cartesian",
      id: "labels",
      data: labels.length ? labels : [],
      getPosition: d => getPosition(d, props.projectionMode),
      updateTriggers: {
        getPosition: [props.projectionMode],
      },
      getText: d => d.label || (d as any).title, // legacy
      getSize: 12,
      sizeUnits: "pixels",
      getColor: [252, 252, 252, 222],
      background: true,
      getBackgroundColor: [10, 10, 10, 80],
      backgroundPadding: [4, 2],
      getTextAnchor: "middle",
      getAlignmentBaseline: "center",
      pickable: true,
      onHover: (info: PickingInfo<LabelPoint>) => props.onLabelHover?.(info.object ?? null, info.object ? [info.x, info.y] : null),
    });
  });


  const mergedConceptPoints = createMemo(() => props.datasets.filter(d => d.type === "concept").flatMap(d => d.points || []));

  const mergedNeighbourPoints = createMemo(() => props.datasets.filter(d => d.type === "concept_neighbours").flatMap(d => d.points || []));

  const mergedClusterPoints = createMemo(() => props.datasets.filter(d => d.type === "concept_clusters").flatMap(d => d.points || []));


  const layers = createMemo(() => {
    const proj = props.projectionMode;
    // Reading zoomTier (not a per-frame zoom value) means this memo only
    // re-runs when pickability flips, not on every scroll tick.
    const zoomed = zoomTier() === 1;

    const conceptPoints = mergedConceptPoints();
    const neighbourPoints = mergedNeighbourPoints();
    const clusterPoints = mergedClusterPoints();

    const layersList: any[] = [];

    if (clusterPoints.length > 0) {
      layersList.push(new GlowScatterplotLayer<PointData>({
        id: "clusters-merged",
        coordinateSystem: "cartesian",
        data: clusterPoints,
        getPosition: p => getPosition(p, proj),
        getFillColor: p => getColor()(p),
        getRadius: 10.0,
        radiusUnits: "pixels",
        opacity: 0.95,
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 100, 180],
        transitions: {
          getPosition: { duration: 600 },
          getFillColor: { duration: 300 },
        },
        updateTriggers: {
          getPosition: [proj],
          getFillColor: [props.colorBy, selectedEventIds()],
        },
        onHover: info => props.onPointHover?.(info.object ?? null, info.object ? [info.x, info.y] : null),
      }));

      layersList.push(
        new TextLayer<LabelPoint>({
          id: "clusters-labels",
          data: clusterPoints,

          coordinateSystem: "cartesian",
          getPosition: p => getPosition(p, proj),

          // prefer label, fallback to description, fallback empty
          getText: (p) => p.label ?? p.description ?? "",

          getSize: 12,
          sizeUnits: "pixels",

          getColor: [255, 255, 255, 220],

          getTextAnchor: "middle",
          getAlignmentBaseline: "center",

          background: true,
          getBackgroundColor: [0, 0, 0, 140],
          backgroundPadding: [4, 2],

          pickable: true,

          updateTriggers: {
            getPosition: [proj],
            getText: [props.colorBy],
          },

          onHover: (info: PickingInfo<PointData>) => {
            props.onPointHover?.(
              info.object ?? null,
              info.object ? [info.x, info.y] : null
            );
          }
        })
      );
    }

    if (conceptPoints.length > 0) {
      layersList.push(
        new GlowScatterplotLayer<PointData>({
          id: "concepts-merged",
          getRadius: 3, // 4.5, // TODO
          coordinateSystem: "cartesian",
          data: conceptPoints,
          getPosition: p => getPosition(p, proj),
          getFillColor: p => getColor()(p),
          radiusUnits: "pixels",
          opacity: 0.96,
          pickable: true,
          autoHighlight: true,
          highlightColor: [255, 255, 255, 80],
          transitions: {
            getPosition: { duration: 600 },
            getFillColor: { duration: 300, easing: (t: number) => t * (2 - t) },
            getRadius: { duration: 200 },
          },
          updateTriggers: {
            getPosition: [proj],
            getFillColor: [props.colorBy, selectedEventIds(), props.colorByFields],
          },
          onHover: info => {
            if (isDragging) return;
            props.onPointHover?.(info.object ?? null, info.object ? [info.x, info.y] : null);
          },
        })
      );
    }

    if (neighbourPoints.length > 0) {
      layersList.push(
        new ScatterplotLayer<PointData>({
          id: "neighbours-merged",
          coordinateSystem: "cartesian",
          data: neighbourPoints,
          getPosition: p => getPosition(p, proj),
          getFillColor: p => getColor()(p, "neighbours"),
          getRadius: p => (p.depth === 2 ? 1.8 : 2.8),
          radiusUnits: "pixels",
          opacity: (props.neighbourOpacity ?? 140) / 255,
          pickable: true, // zoomed,
          autoHighlight: true, // zoomed,
          highlightColor: [255, 255, 255, 100],
          transitions: {
            getPosition: { duration: 450 },
            getFillColor: { duration: 300, easing: (t: number) => t * (2 - t) },
          },
          updateTriggers: {
            getPosition: [proj],
            getFillColor: [props.neighbourOpacity, props.colorBy, selectedEventIds()],
          },
          onHover: info => {
            if (isDragging) return;
            props.onPointHover?.(info.object ?? null, info.object ? [info.x, info.y] : null);
          },
        })
      );
    }

    if (props.bfsDataset?.points?.length && proj === "global" && props.bfsOpacity) {
      layersList.push(
        new GlowScatterplotLayer({
          id: "bfs-global",
          data: props.bfsDataset.points,
          getPosition: getBfsPosition,
          getFillColor: p => {
            const depth = p.depth ?? 0;
            const base = DEPTH_COLORS[depth] || DEPTH_COLORS[2];
            return [base[0], base[1], base[2], props.bfsOpacity ?? 90];
          },
          getRadius: 3.5,
          radiusUnits: "pixels",
          opacity: (props.bfsOpacity ?? 90) / 255,
          pickable: zoomed,
          updateTriggers: {
            getFillColor: [props.bfsOpacity],
          }
        })
      );
    }

    layersList.push(labelLayer());

    return layersList;
  });

  function flyTo(target: [number, number, number], newZoom: number, duration = 800) {
    if (!deck) return;
    deck.setProps({
      initialViewState: {
        target,
        zoom: newZoom,
        minZoom: 2,
        maxZoom: 30,
        transitionDuration: duration,
        transitionInterpolator: new LinearInterpolator(["target", "zoom"]),
      } as OrthographicViewState,
    });
  }

  function handleViewStateChange({ viewState: vs }: { viewState: OrthographicViewState }) {
    const z = (vs.zoom as number) ?? 10;

    // Only flip the tier signal when crossing the threshold — not on every
    // scroll tick — so the layers memo doesn't rebuild every frame.
    const nowZoomedIn = z > ZOOM_THRESHOLD;
    if (nowZoomedIn !== isZoomedIn) {
      isZoomedIn = nowZoomedIn;
      setZoomTier(nowZoomedIn ? 1 : 0);
    }

    if (props.onBoundsChange) {
      const scale = Math.pow(2, z);
      const halfW = (window.innerWidth / 2) / (512 * scale);
      const halfH = (window.innerHeight / 2) / (512 * scale);
      const [cx, cy] = vs.target as [number, number, number];
      props.onBoundsChange({
        minX: cx - halfW,
        maxX: cx + halfW,
        minY: cy - halfH,
        maxY: cy + halfH,
        zoom: z,
      });
    }
  }

  onMount(() => {
    deck = new Deck<OrthographicView>({
      canvas,
      views: new OrthographicView({ id: "ortho", controller: true }),
      initialViewState: INITIAL_VIEW_STATE,
      useDevicePixels: true,
      touchAction: "none",
      layers: [],
      onViewStateChange: handleViewStateChange,
      style: { width: "100%", height: "100%" },
    });

    controller = new SelectionController<PointData>({
      mode: "additive",
      multiKey: "Shift",
    });

    controller.setChangeHandler(set => {
      console.debug("[plot.SelectionController changeHandler]", set);
      const points = set ? currentPoints.filter(p => set.has(p.event_id)) : null;
      props.onSelectionChange?.(points);
    });

    controller.setDragPreview = (rect: ScreenRect | null) => {
      isDragging = rect !== null;
      setDragRect(rect);
      if (rect !== null) {
        props.onPointHover?.(null, null);
      }
    };

    controller
      .use(new DeckClickPlugin(deck, controller))
      .use(new CanvasDragPlugin(canvas, deck, controller));

    // Record pointer-down position for distance-based click vs drag detection.
    canvas.addEventListener("pointerdown", e => {
      pointerDownX = e.offsetX;
      pointerDownY = e.offsetY;
    });

    // Use pointerup rather than click becuase DeckClickPlugin handles  click logic;
    // this handler is only responsible for picking objects and dispatching
    // to the SelectionController.
    // We restore pointerup here because click fires after the drag plugin
    // has already cleared isDragging, making the flag unreliable for filtering
    // drag-end events. pointerup fires before that cleanup.
    canvas.addEventListener("pointerup", async e => {
      // Distance guard: if the pointer travelled more than DRAG_THRESHOLD_PX
      // this is a drag-end, not a click so ignore.
      const dx = e.offsetX - pointerDownX;
      const dy = e.offsetY - pointerDownY;
      if (Math.sqrt(dx * dx + dy * dy) > DRAG_THRESHOLD_PX) return;

      const pick = await deck?.pickObjects({
        x: e.offsetX,
        y: e.offsetY,
      });

      const cleanPick = pick
        ?.filter(p => !p.sourceLayer?.id.startsWith("bfs-"))
        .map(p => p.object)
        .filter((o): o is PointData => !!o?.event_id) ?? [];

      if (cleanPick.length) {
        controller?.dispatch({ type: "click", payload: cleanPick });
      } else {
        controller?.dispatch({ type: "null-select", payload: null });
      }
    });
  });

  createEffect(() => {
    currentPoints = allPoints();
  });

  createEffect(() => {
    deck?.setProps({ layers: layers() });
  });

  // Fit camera to dataset bounds whenever projection mode or datasets change.
  createEffect(() => {
    if (!deck) return;

    flyTo(
      [0.5, 0.5, 0],
      Math.log2(Math.min(canvas.clientWidth, canvas.clientHeight)),
      400
    );
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