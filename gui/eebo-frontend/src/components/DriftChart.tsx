import { createEffect, createSignal, onMount, onCleanup } from "solid-js";
import * as d3 from "d3";
import type { SlicePoint } from "../types";
import styles from "./DriftChart.module.css";

const POINT_RADIUS = 7;
const STROKE_WIDTH = 2;

const MARGIN = {
    top: 10,
    right: 20,
    bottom: 30,
    left: 40
};

type TooltipState = {
    x: number;
    y: number;
    data: any | null;
} | null;

type HistoryEntry = {
    term: string;
    year: number;
};

export default function DriftChart(props: {
    series: Record<string, SlicePoint[]>;
    width?: number;
    height?: number;
    onSelectSlice?: (t: number) => void;
}) {
    let ref: SVGSVGElement | undefined;

    const [size, setSize] = createSignal({ width: 0, height: 0 });
    const [tooltip, setTooltip] = createSignal<TooltipState>(null);
    const [history, setHistory] = createSignal<HistoryEntry[]>([]);
    const [visibleTerms, setVisibleTerms] = createSignal<Set<string>>(new Set());

    const color = d3.scaleOrdinal<string>().range(d3.schemeCategory10);

    const pushHistory = (entry: HistoryEntry) => {
        setHistory(prev => [...prev, entry].slice(-25));
    };

    const toggleTerm = (term: string) => {
        setVisibleTerms(prev => {
            const next = new Set(prev);
            next.has(term) ? next.delete(term) : next.add(term);
            return next;
        });
    };

    onMount(() => {
        if (!ref) return;

        const observer = new ResizeObserver(([entry]) => {
            setSize({
                width: entry.contentRect.width,
                height: entry.contentRect.height
            });
        });

        observer.observe(ref);
        onCleanup(() => observer.disconnect());
    });

    const width = () => props.width ?? size().width;
    const height = () => props.height ?? size().height;

    createEffect(() => {
        const keys = Object.keys(props.series ?? {});
        setVisibleTerms(prev => prev.size ? prev : new Set(keys));
    });

    createEffect(() => {
        const seriesMap = props.series;
        const w = width();
        const h = height();

        if (!seriesMap || !ref || w === 0 || h === 0) return;

        const svg = d3.select(ref);

        // ---------------- CLEAN SLATE EVERY RENDER ----------------
        svg.selectAll("*").remove();

        const termNames = Object.keys(seriesMap);
        const visibleTermsArr = termNames.filter(t => visibleTerms().has(t));

        const visiblePoints = visibleTermsArr.flatMap(term =>
            seriesMap[term].map(p => ({ ...p, term }))
        );

        // ---------------- SCALES ----------------
        const x = d3.scaleLinear()
            .domain(d3.extent(visiblePoints, d => d.slice_start) as [number, number])
            .range([MARGIN.left, w - MARGIN.right]);

        const y = d3.scaleLinear()
            .domain(d3.extent(visiblePoints, d => d.drift) as [number, number])
            .nice()
            .range([h - MARGIN.bottom, MARGIN.top]);

        const line = d3.line<SlicePoint>()
            .x(d => x(d.slice_start))
            .y(d => y(d.drift))
            .curve(d3.curveMonotoneX);

        // ---------------- LINES ----------------
        const linesLayer = svg.append("g").attr("class", styles.driftTermLine);

        linesLayer.selectAll("path")
            .data(visibleTermsArr)
            .enter()
            .append("path")
            .attr("fill", "none")
            .attr("stroke-width", STROKE_WIDTH)
            .attr("stroke", d => color(d) as string)
            .attr("d", d => line(seriesMap[d]));

        // ---------------- POINTS ----------------
        const pointsLayer = svg.append("g").attr("class", styles.driftPointsLayer);

        const pts = visiblePoints.map(d => ({
            ...d,
            sx: x(d.slice_start),
            sy: y(d.drift)
        }));

        pointsLayer.selectAll("circle")
            .data(pts)
            .enter()
            .append("circle")
            .attr("r", POINT_RADIUS)
            .attr("fill", d => color(d.term) as string)
            .attr("cx", d => d.sx)
            .attr("cy", d => d.sy);

        // ---------------- INTERACTION ----------------
        const delaunay = d3.Delaunay.from(
            pts,
            d => d.sx,
            d => d.sy
        );

        const interactionLayer = svg.append("rect")
            .attr("width", w)
            .attr("height", h)
            .attr("fill", "transparent")
            .style("pointer-events", "all");

        interactionLayer
            .on("mousemove", (event) => {
                const [mx, my] = d3.pointer(event);

                const i = delaunay.find(mx, my);
                const d = pts[i];

                if (!d) return setTooltip(null);

                setTooltip({
                    x: mx,
                    y: my,
                    data: d
                });
            })
            .on("mouseleave", () => setTooltip(null))
            .on("click", (event) => {
                const [mx, my] = d3.pointer(event);

                const i = delaunay.find(mx, my);
                const d = pts[i];

                if (!d) return;

                props.onSelectSlice?.(d.slice_start);

                window.dispatchEvent(
                    new CustomEvent("neighbourhood:open", {
                        detail: {
                            term: d.term,
                            year: d.slice_start
                        }
                    })
                );

                // pushHistory({
                //     term: d.term,
                //     year: d.slice_start
                // });
            });
    });

    return (
        <div class={styles.distChartWrapper} style={{ width: "100%", height: "100%" }}>

            {/* LEGEND */}
            <div class={styles.legend}>
                {Object.keys(props.series ?? {}).map(term => (
                    <label>
                        <input
                            type="checkbox"
                            checked={visibleTerms().has(term)}
                            onChange={() => toggleTerm(term)}
                        />
                        <span style={{ color: color(term) }}>
                            {term}
                        </span>
                    </label>
                ))}
            </div>

            {/* HISTORY */}
            <div class={styles.history}>
                {history().map((h, i) => (
                    <div>{i}: {h.term} @ {h.year}</div>
                ))}
            </div>

            {/* TOOLTIP */}
            {tooltip() && (
                <div
                    class={styles.tooltip}
                    style={{
                        left: `${tooltip()!.x + 10}px`,
                        top: `${tooltip()!.y + 10}px`
                    }}
                >
                    <div><b>{tooltip()!.data.term}</b></div>
                    <div>year: {tooltip()!.data.slice_start}</div>
                    <div>drift: {tooltip()!.data.drift}</div>
                </div>
            )}

            <svg
                ref={el => (ref = el)}
                style={{ width: "100%", height: "100%" }}
            />
        </div>
    );
}
