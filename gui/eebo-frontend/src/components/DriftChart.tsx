import { createEffect, createSignal, onMount, onCleanup } from "solid-js";
import * as d3 from "d3";
import type { SlicePoint } from "../types";
import { eeboStore, setEeboStore } from "../stores/Eebo.store";
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

export default function DriftChart(props: {
    series: Record<string, SlicePoint[]>;
    width?: number;
    height?: number;
    onSelectSlice?: (term: string, slice_start: number) => void;
}) {
    let ref: SVGSVGElement | undefined;

    const [size, setSize] = createSignal({ width: 0, height: 0 });
    const [tooltip, setTooltip] = createSignal<TooltipState>(null);
    const [visibleTerms, setVisibleTerms] = createSignal<Set<string>>(new Set());
    const width = () => props.width ?? size().width;
    const height = () => props.height ?? size().height;
    const color = d3.scaleOrdinal<string>().range(d3.schemeCategory10);

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

        // CLEAN SLATE EVERY RENDER
        svg.selectAll("*").remove();

        const termNames = Object.keys(seriesMap);
        const visibleTermsArr = termNames.filter(t => visibleTerms().has(t));

        const visiblePoints = visibleTermsArr.flatMap(term =>
            seriesMap[term].map(p => ({ ...p, term }))
        );

        // SCALES
        const x = d3.scaleLinear()
            .domain(d3.extent(visiblePoints, d => d.slice_start) as [number, number])
            .range([MARGIN.left, w - MARGIN.right]);

        const y = d3.scaleLinear()
            .domain(d3.extent(visiblePoints, d => d.drift) as [number, number])
            .nice()
            .range([
                h - MARGIN.bottom - (POINT_RADIUS * 2),
                MARGIN.top + (POINT_RADIUS * 2)
            ]);

        const line = d3.line<SlicePoint>()
            .x(d => x(d.slice_start))
            .y(d => y(d.drift))
            .curve(d3.curveMonotoneX);

        // LINES
        const linesLayer = svg.append("g").attr("class", styles.driftTermLine);

        linesLayer.selectAll("path")
            .data(visibleTermsArr)
            .enter()
            .append("path")
            .attr("fill", "none")
            .attr("stroke-width", STROKE_WIDTH)
            .attr("stroke", d => color(d) as string)
            .attr("d", d => line(seriesMap[d]));

        // POINTS
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

        // INTERACTION
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
                if (eeboStore.overlay.open) {
                    setEeboStore('overlay', { open: false });
                    console.log('Hid overlay');
                    return;
                }

                const [mx, my] = d3.pointer(event);
                if (!ref) return;
                const rect = ref.getBoundingClientRect();
                const i = delaunay.find(mx, my);
                const d = pts[i];
                console.log('[DriftChart] click found d', d.term, d.slice_start)
                if (!d) return;
                setEeboStore("selected", {
                    token: d.term,
                    slice_start: d.slice_start,
                    slice_end: d.slice_end ?? d.slice_start,
                    color: color(d.term) as string,
                });
                setEeboStore('overlay', {
                    x: rect.left + mx,
                    y: rect.top + my,
                    open: true,
                });
                props.onSelectSlice?.(d.term, Number(d.slice_start));

                // pushHistory({
                //     term: d.term,
                //     year: d.slice_start
                // });
            });
    });

    return (
        <article
            classList={{
                [styles.distChartWrapper]: true,
                [styles.dimmed]: eeboStore.overlay.open,
                [styles.disabled]: eeboStore.overlay.open
            }}
            style={{ width: "100%", height: "100%" }}
        >

            {/* LEGEND */}
            <header class={'responsive surface-container-high border ' + styles.driftLegend}>
                <div class="field middle-align">
                    <nav class="wrap padding middle-align">
                        {Object.keys(props.series ?? {}).map(term => (
                            <button class="chip">
                                <label class="checkbox small">
                                    <input type="checkbox"
                                        checked={visibleTerms().has(term)}
                                        onChange={() => toggleTerm(term)}
                                    />
                                    <span style={{ color: color(term) }} class={styles.driftLegendItem}>
                                        {term}
                                    </span>
                                </label>
                            </button>
                        ))}
                    </nav>
                </div>
            </header>

            {/* TOOLTIP */}
            {
                tooltip() && (
                    <aside
                        class={styles.driftTooltip}
                        style={{
                            left: `${tooltip()!.x + 10}px`,
                            top: `${tooltip()!.y + 10}px`
                        }}
                    >
                        <h6 class="bottom-margin">{tooltip()!.data.term}</h6>
                        <div>Year: {tooltip()!.data.slice_start}</div>
                        <div>Drift: {tooltip()!.data.drift}</div>
                    </aside>
                )
            }

            <svg
                ref={el => (ref = el)}
                style={{ width: "100%", height: "100%" }}
            />
        </article >
    );
}
