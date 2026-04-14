import { createEffect, createSignal, createMemo, onCleanup, onMount, Show } from "solid-js";
import * as d3 from "d3";

import type { SlicePoint, NamedSlicePoint } from "../types";
import { eeboStore, setNullSelected } from "../stores/Eebo.store";
import DriftLegend from "./DriftLegend";
import styles from "./DriftChart.module.css";
import SLICE_RANGES from "../services/SLICES.json";

export const color = d3.scaleOrdinal<string>().range(d3.schemeCategory10);

const POINT_RADIUS = 7;

const MARGIN = {
    top: 5,
    right: 5,
    bottom: 5,
    left: 5
};

type ScreenPoint = NamedSlicePoint & {
    sx: number;
    sy: number;
};

type TooltipState = {
    x: number;
    y: number;
    data: ScreenPoint;
} | null;

type Props = {
    series: Record<string, SlicePoint[]>;
    width?: number;
    height?: number;
    onSelectSlice?: (d: ScreenPoint) => void;
}

export default function DriftChart(props: Props) {
    let svgRef: SVGSVGElement | undefined;

    const [size, setSize] = createSignal({ width: 0, height: 0 });
    const [tooltip, setTooltip] = createSignal<TooltipState>(null);
    const [visibleTerms, setVisibleTerms] = createSignal<Set<string>>(new Set());

    const terms = () => Object.keys(props.series ?? {});


    const width = () => props.width ?? size().width;
    const height = () => props.height ?? size().height;


    const setAll = () => new Set(terms());
    const setSolo = (t: string) => new Set([t]);

    const isActiveMode = () => eeboStore.selected.token;

    const activeRange = createMemo(() => {
        return SLICE_RANGES[eeboStore.sliceIndex];
    });

    const inActiveRange = (p: NamedSlicePoint) => {
        if (!isActiveMode()) return false;

        const r = activeRange();
        if (!r) return false;

        const [a, b] = r;
        return p.slice_start >= a && p.slice_start <= b;
    };

    const groupScore = (pts: NamedSlicePoint[]) => {
        if (!isActiveMode()) return 0;

        const range = activeRange();
        if (!range) return 0;

        const [a, b] = range;

        let hit = 0;
        for (const p of pts) {
            if (p.slice_start >= a && p.slice_start <= b) hit++;
        }

        return hit / pts.length;
    };

    const toggleOne = (prev: Set<string>, term: string) => {
        const next = new Set(prev);
        next.has(term) ? next.delete(term) : next.add(term);
        return next.size ? next : setAll();
    };

    let clickTimer: number | undefined;

    const handleClick = (term: string) => {
        if (clickTimer !== undefined) {
            clearTimeout(clickTimer);
            clickTimer = undefined;

            const current = visibleTerms();

            setVisibleTerms(
                current.size === 1 && current.has(term)
                    ? setAll()
                    : setSolo(term)
            );
            return;
        }

        clickTimer = window.setTimeout(() => {
            setVisibleTerms(prev => toggleOne(prev, term));
            clickTimer = undefined;
        }, 250);
    };

    onCleanup(() => {
        if (clickTimer) clearTimeout(clickTimer);
    });

    // resize observer
    onMount(() => {
        if (!svgRef) return;

        const ro = new ResizeObserver(([entry]) => {
            setSize({
                width: entry.contentRect.width,
                height: entry.contentRect.height
            });
        });

        ro.observe(svgRef);
        onCleanup(() => ro.disconnect());
    });

    // init visibility
    createEffect(() => {
        const keys = terms();
        setVisibleTerms(prev => prev.size ? prev : new Set(keys));
    });

    const visibleData = createMemo(() => {
        const vis = visibleTerms();
        const seriesMap = props.series ?? {};

        const out: ScreenPoint[] = [];

        for (const term of Object.keys(seriesMap)) {
            if (!vis.has(term)) continue;

            for (const p of seriesMap[term]) {
                out.push({
                    ...p,
                    term,
                    sx: 0,
                    sy: 0
                });
            }
        }

        return out;
    });

    const geom = createMemo(() => {
        const data = visibleData();
        const w = width();
        const h = height();

        if (!data.length || !w || !h) return null;

        const x = d3.scaleLinear()
            .domain(d3.extent(data, d => d.slice_start) as [number, number])
            .range([MARGIN.left, w - MARGIN.right]);

        const y = d3.scaleLinear()
            .domain(d3.extent(data, d => d.drift) as [number, number])
            .nice()
            .range([h - MARGIN.bottom - POINT_RADIUS * 2, MARGIN.top + POINT_RADIUS * 2]);

        const pts = data.map(d => ({
            ...d,
            sx: x(d.slice_start),
            sy: y(d.drift)
        }));

        const delaunay = d3.Delaunay.from(
            pts,
            d => d.sx,
            d => d.sy
        );

        const line = d3.line<NamedSlicePoint>()
            .x(d => x(d.slice_start))
            .y(d => y(d.drift))
            .curve(d3.curveMonotoneX);

        return { x, y, pts, delaunay, line };
    });

    const handleMove = (event: MouseEvent) => {
        if (eeboStore.selected.token) return;

        const g = geom();
        if (!g) return setTooltip(null);

        const [mx, my] = d3.pointer(event);
        const i = g.delaunay.find(mx, my);
        const d = g.pts[i];

        if (!d) return setTooltip(null);

        setTooltip({ x: mx, y: my, data: d });
    };

    const handleLeave = () => setTooltip(null);

    const handleClickSvg = (event: MouseEvent) => {
        if (eeboStore.selected.token) {
            setNullSelected();
        }
        const g = geom();
        if (!g) return;
        setTooltip(null);

        const [mx, my] = d3.pointer(event);
        const i = g.delaunay.find(mx, my);
        const d = g.pts[i];

        if (!d) return;
        props.onSelectSlice?.(d);
    };

    return (
        <article
            classList={{
                [styles.driftChartWrapper]: true,
                [styles.dimmed]: eeboStore.selected.token !== null
            }}
        >
            {/* TOOLTIP */}
            <Show when={tooltip()}>
                <aside
                    class={styles.driftTooltip + ' surface-container-high'}
                    style={{
                        left: `${tooltip()!.x + 10}px`,
                        top: `${tooltip()!.y + 10}px`
                    }}
                >
                    <h6>{tooltip()!.data.term}</h6>
                    <div>Year: {tooltip()!.data.slice_start}</div>
                    <div>Drift: {tooltip()!.data.drift}</div>
                </aside>
            </Show>

            {/* SVG */}
            <svg
                ref={el => (svgRef = el)}
                style={{ width: "100%", height: "100%", position: "absolute", top: 0, left: 0 }}
                onMouseMove={handleMove}
                onMouseLeave={handleLeave}
                onClick={handleClickSvg}
            >
                {(() => {
                    const g = geom();
                    if (!g) return null;

                    const groups = new Map<string, NamedSlicePoint[]>();

                    for (const p of g.pts) {
                        if (!groups.has(p.term)) groups.set(p.term, []);
                        groups.get(p.term)!.push(p);
                    }

                    return (
                        <>
                            {/* LINES */}
                            <g>
                                {Array.from(groups.entries()).map(([term, pts]) => {
                                    const intensity = groupScore(pts);

                                    return (
                                        <path
                                            d={g.line(pts)!}
                                            fill="none"
                                            stroke={color(term)}
                                            stroke-width={isActiveMode() ? 1 + intensity * 3 : 2}
                                            opacity={isActiveMode() ? 0.2 + intensity * 0.8 : 1}
                                        />
                                    );
                                })}
                            </g>

                            {/* POINTS */}
                            <g>
                                {g.pts.map(p => (
                                    <circle cx={p.sx} cy={p.sy} r={POINT_RADIUS}
                                        class={p.term === eeboStore.selected.token ? styles.selectedTermNode : ''}
                                        fill={color(p.term)}
                                        stroke-color={'white'}
                                        opacity={isActiveMode() ? (inActiveRange(p) ? 1 : 0.25) : 1}
                                    />
                                ))}
                            </g>
                        </>
                    );
                })()}
            </svg>

            {/* LEGEND (EXTRACTED) */}
            <DriftLegend
                terms={terms()}
                visible={visibleTerms()}
                onToggle={handleClick}
            />
        </article>
    );
}