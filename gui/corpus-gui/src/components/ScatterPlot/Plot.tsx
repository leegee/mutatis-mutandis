// ScatterPlot/Plot.tsx
// Full-screen WebGL scatter plot for Tier 3 UMAP event data.
// Pure render component: all state lives in the parent.

import { Deck, LinearInterpolator, OrthographicView, type OrthographicViewState } from "@deck.gl/core";
import { TextLayer } from "@deck.gl/layers";
import { createEffect, createMemo, createSignal, onCleanup, onMount } from "solid-js";
import { controlsActions } from "~/state/controls.actions";

import { buildColorMap } from "../../lib/colour";
import { GlowScatterplotLayer } from "./GlowScatterplotLayer";
import { CanvasDragPlugin } from "./SelectionPlugin/CanvasDragPlugin";
import { DeckClickPlugin } from "./SelectionPlugin/DeckClickPlugin";
import { SelectionController } from "./SelectionPlugin/SelectionController";
import type { ScreenRect } from "./SelectionPlugin/types";

import type { UmapCluster, UmapDataset, UmapPoint } from "./types";

import "./style.css";

type RGB = [number, number, number];
type RGBA = [number, number, number, number];

const DRAG_THRESHOLD_PX = 6;

const GREY: RGBA = [120, 120, 130, 140];

const INITIAL_VIEW_STATE: OrthographicViewState = {
  target: [0, 0, 0],
  zoom: 10,
  minZoom: -10,
  maxZoom: 100,
};

const brighten = ([r, g, b]: RGB | RGBA): RGBA => [
  Math.min(r * 1.15, 255),
  Math.min(g * 1.15, 255),
  Math.min(b * 1.15, 255),
  255,
];

const dim = ([r, g, b, a]: RGBA): RGBA => [r * 0.75, g * 0.75, b * 0.75, a];

interface PlotProps {
  dataset: UmapDataset;
  pointRadius?: number;
  selectedEventIds?: Set<string>;
  plotPointScaleFactor: number;
  showClusterCentroids?: boolean;
  onPointHover?: (point: UmapPoint | UmapCluster | null, screenXY: [number, number] | null) => void;
  onSelectionChange?: (eventIds: Set<string>) => void;
}

const getPosition = (point: UmapPoint | UmapCluster): [number, number, number] => [point.x, point.y, 0];

function getDatasetBounds(points: UmapPoint[]) {
  if (!points.length) return;

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  for (const point of points) {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      continue;
    }

    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  }

  if (!Number.isFinite(minX)) return;

  return {
    cx: (minX + maxX) / 2,
    cy: (minY + maxY) / 2,
    extent: Math.max(maxX - minX, maxY - minY),
  };
}

export default function Plot(props: PlotProps) {
  let canvas!: HTMLCanvasElement;
  let deck: Deck<OrthographicView> | null = null;
  let controller: SelectionController<UmapPoint> | undefined;
  let currentPoints: UmapPoint[] = [];
  const [fontFamily, setFontFamily] = createSignal("sans-serif");

  let pointerDownX = 0;
  let pointerDownY = 0;

  let isDragging = false;
  const [dragRect, setDragRect] = createSignal<ScreenRect | null>(null);

  const allPoints = createMemo(() => props.dataset.points);

  const selectedEventIds = createMemo(() => props.selectedEventIds ?? new Set<string>());

  const colorMap = createMemo(() => buildColorMap(allPoints().map((point) => String(point.clusterId ?? ""))));

  const getPointColor = createMemo(() => {
    return (point: UmapPoint): RGBA => {
      const base = colorMap().get(String(point.clusterId ?? "")) ?? GREY;

      const selected = selectedEventIds();

      if (!selected.size) return base;

      if (selected.has(point.eventId)) {
        return brighten(base);
      }

      return dim(base);
    };
  });

  const layers = createMemo(() => {
    const points = allPoints();
    const clusters = props.showClusterCentroids ? props.dataset.clusters : [];
    const pointScale = props.plotPointScaleFactor;
    const pointRadius = props.pointRadius ?? 5;

    // These layers deliberately share the same x/y coordinate system.
    const layersList = [];

    if (clusters.length > 0) {
      layersList.push(
        new GlowScatterplotLayer<UmapCluster>({
          id: "clusters",
          coordinateSystem: "cartesian",
          data: clusters,
          getPosition,
          getFillColor: (cluster) => colorMap().get(String(cluster.clusterId)) ?? GREY,
          getRadius: 10 * pointScale,
          radiusUnits: "pixels",
          opacity: 0.25,
          pickable: true,
          autoHighlight: true,
          highlightColor: [255, 255, 100, 180],
          transitions: {
            getPosition: { duration: 300 },
            getFillColor: { duration: 300 },
          },
          updateTriggers: {
            getRadius: [props.plotPointScaleFactor],
            getFillColor: [props.dataset, selectedEventIds()],
          },
          onHover: (info) => {
            if (isDragging) return;

            const cluster = info.object ?? null;

            props.onPointHover?.(cluster, cluster ? [info.x, info.y] : null);
          },
        }),
      );

      layersList.push(
        new TextLayer<UmapCluster>({
          id: "cluster-labels",
          data: clusters,
          coordinateSystem: "cartesian",
          getPosition,
          getSize: 12,
          getPixelOffset: [0, 20],
          fontFamily: fontFamily(),
          fontWeight: "bold",
          fontSettings: {
            sdf: true,
          },
          getText: (cluster) => cluster.label ?? "",
          sizeUnits: "pixels",
          getColor: [255, 255, 255, 220],
          getTextAnchor: "middle",
          getAlignmentBaseline: "center",
          background: true,
          getBackgroundColor: [0, 0, 0, 140],
          backgroundPadding: [4, 2],
          pickable: true,
          onHover: (info) => {
            if (isDragging) return;

            const cluster = info.object ?? null;

            props.onPointHover?.(cluster, cluster ? [info.x, info.y] : null);
          },
        }),
      );
    }

    if (points.length > 0) {
      layersList.push(
        new GlowScatterplotLayer<UmapPoint>({
          id: "points",
          coordinateSystem: "cartesian",
          data: points,
          getPosition,
          getFillColor: (point) => getPointColor()(point),
          radiusUnits: "pixels",
          getRadius: pointRadius * pointScale,
          opacity: 0.96,
          pickable: true,
          autoHighlight: true,
          highlightColor: [255, 255, 255, 80],
          transitions: {
            getPosition: { duration: 300 },
            getFillColor: {
              duration: 300,
              easing: (t: number) => t * (2 - t),
            },
            getRadius: { duration: 200 },
          },
          updateTriggers: {
            getRadius: [props.plotPointScaleFactor, props.pointRadius],
            getFillColor: [props.dataset, props.selectedEventIds],
          },
          onHover: (info) => {
            if (isDragging) return;

            const point = info.object ?? null;

            props.onPointHover?.(point, point ? [info.x, info.y] : null);
          },
        }),
      );
    }

    return layersList;
  });

  function flyTo(target: [number, number, number], newZoom: number, duration = 800) {
    if (!deck) return;

    deck.setProps({
      initialViewState: {
        target,
        zoom: Math.max(INITIAL_VIEW_STATE.minZoom as number, Math.min(INITIAL_VIEW_STATE.maxZoom as number, newZoom)),
        minZoom: INITIAL_VIEW_STATE.minZoom,
        maxZoom: INITIAL_VIEW_STATE.maxZoom,
        transitionDuration: duration,
        transitionInterpolator: new LinearInterpolator(["target", "zoom"]),
      } as OrthographicViewState,
    });
  }

  onMount(() => {
    setFontFamily(window.getComputedStyle(document.body).fontFamily);

    deck = new Deck<OrthographicView>({
      canvas,
      views: new OrthographicView({
        id: "ortho",
        controller: true,
      }),
      initialViewState: INITIAL_VIEW_STATE,
      useDevicePixels: true,
      touchAction: "none",
      layers: [],
      style: {
        width: "100%",
        height: "100%",
      },
    });

    controller = new SelectionController<UmapPoint>({
      mode: "additive",
      multiKey: "Shift",
    });

    controller.setChangeHandler((set) => {
      // const points = set ? currentPoints.filter((point) => set.has(point.eventId)) : null;
      // props.onSelectionChange?.(points);
      props.onSelectionChange?.(set ?? new Set<string>());
    });

    controller.setDragPreview = (rect: ScreenRect | null) => {
      isDragging = rect !== null;
      setDragRect(rect);

      if (rect !== null) {
        props.onPointHover?.(null, null);
      }
    };

    controller.use(new DeckClickPlugin(deck, controller)).use(new CanvasDragPlugin(canvas, deck, controller));

    canvas.addEventListener("pointerdown", (event) => {
      pointerDownX = event.offsetX;
      pointerDownY = event.offsetY;
    });

    canvas.addEventListener("pointerup", async (event) => {
      const dx = event.offsetX - pointerDownX;
      const dy = event.offsetY - pointerDownY;

      if (Math.sqrt(dx * dx + dy * dy) > DRAG_THRESHOLD_PX) {
        return;
      }

      const pick = await deck?.pickObjects({
        x: event.offsetX,
        y: event.offsetY,
      });

      const pickedPoints =
        pick
          ?.map((result) => result.object)
          .filter((object): object is UmapPoint => !!object && typeof object.eventId === "string") ?? [];

      if (pickedPoints.length) {
        controller?.dispatch({
          type: "click",
          payload: pickedPoints,
        });
      } else {
        controller?.dispatch({
          type: "null-select",
          payload: null,
        });
      }
    });
  });

  createEffect(() => {
    currentPoints = allPoints();
  });

  createEffect(() => {
    deck?.setProps({
      layers: layers(),
    });
  });

  createEffect(() => {
    const points = allPoints();

    if (!points.length || !deck) return;

    const fit = getDatasetBounds(points);

    if (!fit) return;

    const padding = 1.2;
    const canvasSize = Math.min(canvas.clientWidth, canvas.clientHeight);

    if (canvasSize <= 0 || fit.extent <= 0) return;

    const zoom = Math.max(
      INITIAL_VIEW_STATE.minZoom as number,
      Math.min(INITIAL_VIEW_STATE.maxZoom as number, Math.log2(canvasSize / (fit.extent * padding))),
    );

    flyTo([fit.cx, fit.cy, 0], zoom, 400);
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
