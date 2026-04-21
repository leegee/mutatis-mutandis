import { createEffect, createSignal, createMemo, onCleanup, onMount, Show } from "solid-js";
import * as d3 from "d3";

import type { SlicePoint } from "../types";
import { eeboStore, setNullSelected } from "../stores/Eebo.store";
import DriftLegend from "./DriftLegend";
import styles from "./DriftChart.module.css";
import SLICE_RANGES from "../services/SLICES.json";

export const color = d3.scaleOrdinal<string, string>();

type ScreenPoint = SlicePoint & {
    sx: number;
    sy: number;
    term: string;
};

type TooltipState = {
    x: number;
    y: number;
    data: ScreenPoint;
} | null;

type Props = {
    series: Record<string, Record<string, SlicePoint>>;
    width?: number;
    height?: number;
    onSelectSlice?: (d: ScreenPoint) => void;
};

export default function DriftChart(props: Props) {
    let svgRef: SVGSVGElement | undefined;

    const [size, setSize] = createSignal({ width: 0, height: 0 });
    const [tooltip, setTooltip] = createSignal<TooltipState>(null);
    const [visibleTerms, setVisibleTerms] = createSignal<Set<string>>(new Set());

    const width = () => props.width ?? size().width;
    const height = () => props.height ?? size().height;

    const terms = () => Object.keys(props.series ?? {});

    const setAll = () => new Set(terms());
    const setSolo = (t: string) => new Set([t]);

    const isActiveMode = () => eeboStore.selected.token;

    const activeRange = createMemo(() => {
        return SLICE_RANGES[eeboStore.sliceIndex];
    });

    const colorScale = createMemo(() => {
        const t = [...terms()].sort();
        const n = t.length;

        if (n <= 10) {
            return d3.scaleOrdinal<string, string>()
                .domain(t)
                .range(d3.schemeTableau10);
        }

        return d3.scaleOrdinal<string, string>()
            .domain(t)
            .range(t.map((_, i) =>
                d3.hsl((i * 360) / n, 0.65, 0.55).formatHex()
            ));
    });

    const inActiveRange = (p: ScreenPoint) => {
        if (!isActiveMode()) return false;
        const r = activeRange();
        if (!r) return false;

        const [a, b] = r;
        return p.slice_start >= a && p.slice_start <= b;
    };

    const groupScore = (pts: ScreenPoint[]) => {
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
            setVisibleTerms(prev => {
                const next = new Set(prev);
                next.has(term) ? next.delete(term) : next.add(term);
                return next.size ? next : setAll();
            });
            clickTimer = undefined;
        }, 250);
    };

    onCleanup(() => {
        if (clickTimer) clearTimeout(clickTimer);
    });

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

            const sliceMap = seriesMap[term];

            // enforce chronological order
            const orderedKeys = Object.keys(sliceMap)
                .sort((a, b) => {
                    const [aStart] = a.split("-").map(Number);
                    const [bStart] = b.split("-").map(Number);
                    return aStart - bStart;
                });

            for (const sliceKey of orderedKeys) {
                const p = sliceMap[sliceKey];

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
            .range([5, w - 5]);

        // stabilized Y scale for OT drift
        const yMax = d3.max(data, d => d.drift) ?? 1;

        const y = d3.scaleLinear()
            .domain([0, yMax * 1.1])
            .range([h - 5, 5]);

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

        const line = d3.line<ScreenPoint>()
            .x(d => x(d.slice_start))
            .y(d => y(d.drift))
            .curve(d3.curveMonotoneX);

        return { x, y, pts, delaunay, line };
    });

    const handleMove = (event: MouseEvent) => {
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
        if (eeboStore.selected.token) setNullSelected();

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
        <article classList={{
            [styles.driftChartWrapper]: true,
            [styles.dimmed]: eeboStore.selected.token !== null
        }}>
            <DriftLegend
                terms={terms()}
                visible={visibleTerms()}
                onToggle={handleClick}
                colorScale={colorScale()}
            />

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
                    <div>Drift: {tooltip()!.data.drift.toFixed(4)}</div>
                </aside>
            </Show>

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

                    const groups = new Map<string, ScreenPoint[]>();

                    for (const p of g.pts) {
                        if (!groups.has(p.term)) groups.set(p.term, []);
                        groups.get(p.term)!.push(p);
                    }

                    return (
                        <>
                            <g>
                                {Array.from(groups.entries()).map(([term, pts]) => {
                                    const intensity = groupScore(pts);

                                    // ensure ordered lines
                                    const sorted = pts.slice()
                                        .sort((a, b) => a.slice_start - b.slice_start);

                                    return (
                                        <path
                                            d={g.line(sorted)!}
                                            fill="none"
                                            stroke={colorScale()(term)}
                                            stroke-width={isActiveMode() ? 1 + intensity * 3 : 2}
                                            opacity={isActiveMode() ? 0.5 + intensity * 0.8 : 1}
                                        />
                                    );
                                })}
                            </g>

                            <g>
                                {g.pts.map(p => (
                                    <circle
                                        cx={p.sx}
                                        cy={p.sy}
                                        r={7}
                                        fill={colorScale()(p.term)}
                                        opacity={isActiveMode() ? (inActiveRange(p) ? 1 : 0.25) : 1}
                                    />
                                ))}
                            </g>
                        </>
                    );
                })()}
            </svg>
        </article>
    );
}
