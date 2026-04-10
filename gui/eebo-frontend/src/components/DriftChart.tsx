import { createEffect, createSignal, onMount, onCleanup } from "solid-js";
import * as d3 from "d3";
import type { SlicePoint } from "../types";
import styles from "./DriftChart.module.css";

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

    const color = d3.scaleOrdinal<string>().range(d3.schemeCategory10);

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
        setVisibleTerms(new Set(Object.keys(props.series ?? {})));
    });

    createEffect(() => {
        const seriesMap = props.series;
        const w = width();
        const h = height();
        if (!seriesMap || !ref || w === 0 || h === 0) return;

        const svg = d3.select(ref);

        const termNames = Object.keys(seriesMap);
        const visibleTermsArr = termNames.filter(t => visibleTerms().has(t));

        const allPoints = visibleTermsArr.flatMap(term =>
            seriesMap[term].map(p => ({ ...p, term }))
        );

        const x = d3.scaleLinear()
            .domain(d3.extent(allPoints, d => d.slice_start) as [number, number])
            .range([MARGIN.left, w - MARGIN.right]);

        const y = d3.scaleLinear()
            .domain(d3.extent(allPoints, d => d.drift) as [number, number])
            .nice()
            .range([h - MARGIN.bottom, MARGIN.top]);

        const line = d3.line<SlicePoint>()
            .x(d => x(d.slice_start))
            .y(d => y(d.drift))
            .curve(d3.curveMonotoneX);

        // ---------------- LINES ----------------
        const paths = svg.selectAll<SVGPathElement, string>("path.term-line")
            .data(visibleTermsArr, d => d);

        paths.enter()
            .append("path")
            .attr("class", styles.driftTermLine)
            .merge(paths as any)
            .attr("fill", "none")
            .attr("stroke", d => color(d) as string)
            .attr("d", d => line(seriesMap[d]))
            .attr("opacity", 1);

        paths.exit().remove();

        // ---------------- POINTS (visual only) ----------------
        const points = svg.selectAll<SVGCircleElement, any>("circle.drift-point")
            .data(allPoints, d => `${d.term}-${d.slice_start}`);

        points.enter()
            .append("circle")
            .attr("class", "drift-point")
            .attr("r", 10)
            .attr("opacity", 0.7)
            .attr("fill", d => color(d.term) as string)
            .style("pointer-events", "none")
            .merge(points as any)
            .attr("cx", d => x(d.slice_start))
            .attr("cy", d => y(d.drift));

        points.exit().remove();

        // ---------------- DATA FOR INTERACTION ----------------
        const pts = allPoints.map(d => ({
            ...d,
            sx: x(d.slice_start),
            sy: y(d.drift)
        }));

        const delaunay = d3.Delaunay.from(
            pts,
            d => d.sx,
            d => d.sy
        );

        // ---------------- SAFE HIT LAYER (FIX) ----------------
        const hitLayer = svg.selectAll<SVGRectElement, any>("rect.hit-layer")
            .data([0]);

        hitLayer.enter()
            .append("rect")
            .attr("class", "hit-layer")
            .merge(hitLayer as any)
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", w)
            .attr("height", h)
            .style("fill", "transparent")
            .style("pointer-events", "all")
            .on("mousemove", (event) => {
                const [mx, my] = d3.pointer(event);

                const i = delaunay.find(mx, my);
                const d = pts[i];
                if (!d) return setTooltip(null);

                setTooltip({ x: mx, y: my, data: d });
            })
            .on("mouseleave", () => {
                setTooltip(null);
            })
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

                pushHistory({
                    term: d.term,
                    year: d.slice_start
                });
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
