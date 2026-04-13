import { createEffect, createSignal, onCleanup, onMount } from "solid-js";
import * as d3 from "d3";
import type { SlicePoint, NamedSlicePoint } from "../types";
import { closeOverlay, eeboStore } from "../stores/Eebo.store";
import styles from "./DriftChart.module.css";

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

export default function DriftChart(props: {
    series: Record<string, SlicePoint[]>;
    width?: number;
    height?: number;
    onSelectSlice?: (d: ScreenPoint, x: number, y: number) => void;
}) {
    let svgRef: SVGSVGElement | undefined;
    let interactionRef: SVGRectElement | undefined;

    const [size, setSize] = createSignal({ width: 0, height: 0 });
    const [tooltip, setTooltip] = createSignal<TooltipState>(null);
    const [visibleTerms, setVisibleTerms] = createSignal<Set<string>>(new Set());

    const width = () => props.width ?? size().width;
    const height = () => props.height ?? size().height;

    // ----------------------------
    // visibility helpers (pure)
    // ----------------------------
    const allTerms = () => Object.keys(props.series ?? {});

    const setAll = () => new Set(allTerms());
    const setSolo = (t: string) => new Set([t]);

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

    // ----------------------------
    // resize observer
    // ----------------------------
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

    // ----------------------------
    // init visibility
    // ----------------------------
    createEffect(() => {
        const keys = allTerms();
        setVisibleTerms(prev => prev.size ? prev : new Set(keys));
    });

    // ----------------------------
    // derived dataset (Solid owns data)
    // ----------------------------
    const visibleData = () => {
        const seriesMap = props.series;
        const vis = visibleTerms();

        const out: ScreenPoint[] = [];

        for (const term of Object.keys(seriesMap ?? {})) {
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
    };

    // ----------------------------
    // D3 MATH ONLY (scales + geometry)
    // ----------------------------
    const geom = () => {
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
    };

    // ----------------------------
    // interaction handlers (stable refs)
    // ----------------------------
    const handleMove = (event: MouseEvent) => {
        if (eeboStore._overlay.open) return;

        const g = geom();
        if (!g) return setTooltip(null);

        const [mx, my] = d3.pointer(event);
        const i = g.delaunay.find(mx, my);
        const d = g.pts[i];

        if (!d) return setTooltip(null);

        setTooltip({ x: mx, y: my, data: d });
    };

    const handleLeave = () => setTooltip(null);

    const handleSvgClick = (event: MouseEvent) => {
        if (eeboStore._overlay.open) {
            closeOverlay();
            return;
        }

        const g = geom();
        if (!g || !tooltip()) return;

        const [mx, my] = d3.pointer(event);
        const i = g.delaunay.find(mx, my);
        const d = g.pts[i];
        if (!d) return;

        const cx = mx - width() / 2;
        const cy = my - height() / 2;

        props.onSelectSlice?.(d, cx, cy);
    };

    // ----------------------------
    // RENDER
    // ----------------------------
    return (
        <article
            classList={{
                [styles.driftChartWrapper]: true,
                [styles.dimmed]: eeboStore._overlay.open,
                [styles.disabled]: eeboStore._overlay.open
            }}
            style={{ width: "100%", height: "100%" }}
        >
            {/* LEGEND */}
            <header class={'responsive surface-container-high border ' + styles.driftLegend}>
                <nav class="wrap padding middle-align">
                    {Object.keys(props.series ?? {}).map(term => (
                        <label
                            class="chip checkbox small"
                            onClick={(e) => {
                                e.preventDefault();
                                handleClick(term);
                            }}
                        >
                            <input
                                type="checkbox"
                                checked={visibleTerms().has(term)}
                                onChange={() => { }}
                            />
                            <span style={{ color: color(term) }}>
                                {term}
                            </span>
                        </label>
                    ))}
                </nav>
            </header>

            {/* TOOLTIP */}
            {tooltip() && (
                <aside
                    class={styles.driftTooltip}
                    style={{
                        left: `${tooltip()!.x + 10}px`,
                        top: `${tooltip()!.y + 10}px`
                    }}
                >
                    <h6>{tooltip()!.data.term}</h6>
                    <div>Year: {tooltip()!.data.slice_start}</div>
                    <div>Drift: {tooltip()!.data.drift}</div>
                </aside>
            )}

            {/* SVG = PURE RENDER TARGET */}
            <svg
                ref={el => (svgRef = el)}
                style={{ width: "100%", height: "100%" }}
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
                                {Array.from(groups.entries()).map(([term, pts]) => (
                                    <path
                                        d={g.line(pts)!}
                                        fill="none"
                                        stroke={color(term)}
                                        stroke-width="2"
                                    />
                                ))}
                            </g>

                            {/* POINTS */}
                            <g>
                                {g.pts.map(p => (
                                    <circle
                                        cx={p.sx}
                                        cy={p.sy}
                                        r={POINT_RADIUS}
                                        fill={color(p.term)}
                                    />
                                ))}
                            </g>

                            {/* INTERACTION LAYER */}
                            <rect
                                ref={el => (interactionRef = el)}
                                width={width()}
                                height={height()}
                                fill="transparent"
                                onMouseMove={handleMove}
                                onMouseLeave={handleLeave}
                                onClick={handleSvgClick}
                            />
                        </>
                    );
                })()}
            </svg>
        </article>
    );
}
