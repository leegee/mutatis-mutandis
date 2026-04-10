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

export type SliceView = {
    token: string;
    slices: SlicePoint[];
};

export type DriftSeriesMap = Record<string, SliceView>;

interface DriftChartProps {
    series: Record<string, SlicePoint[]>;
    width?: number;
    height?: number;
    onSelectSlice?: (t: number) => void;
}

export default function DriftChart(props: DriftChartProps) {
    let ref: SVGSVGElement | undefined;

    const [size, setSize] = createSignal({ width: 0, height: 0 });

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

    // ---------------- visibility ----------------
    const [visibleTerms, setVisibleTerms] = createSignal<Set<string>>(new Set());

    const toggleTerm = (term: string) => {
        setVisibleTerms(prev => {
            const next = new Set(prev);
            next.has(term) ? next.delete(term) : next.add(term);
            return next;
        });
    };

    const color = d3.scaleOrdinal<string>()
        .range(d3.schemeCategory10);

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

        const visibleTermsArr = termNames.filter(t =>
            visibleTerms().has(t)
        );

        const allX = termNames.flatMap(term =>
            seriesMap[term].map(s => s.slice_start)
        );

        const allY = termNames.flatMap(term =>
            seriesMap[term].map(s => s.drift)
        );

        const x = d3.scaleLinear()
            .domain(d3.extent(allX) as [number, number])
            .range([MARGIN.left, w - MARGIN.right]);

        const y = d3.scaleLinear()
            .domain(d3.extent(allY) as [number, number])
            .nice()
            .range([h - MARGIN.bottom, MARGIN.top]);

        const line = d3.line<SlicePoint>()
            .x(d => x(d.slice_start))
            .y(d => y(d.drift))
            .curve(d3.curveMonotoneX);

        const paths = svg.selectAll<SVGPathElement, string>(".term-line")
            .data(visibleTermsArr, d => d);

        paths.enter()
            .append("path")
            .attr("class", styles.driftTermLine)
            .attr("stroke", d => color(d) as string)
            .merge(paths as any)
            .attr("d", d => line(seriesMap[d]))
            .transition()
            .duration(250)
            .attr("opacity", 1);

        paths.exit()
            .transition()
            .duration(200)
            .attr("opacity", 0)
            .remove();

        // ---------------- current point ----------------
        svg.selectAll(".current-point").remove();

        const firstTerm = termNames[0];
        const lastSlice = seriesMap[firstTerm]?.at(-1);

        if (lastSlice) {
            svg.append("circle")
                .attr("class", "current-point")
                .attr("cx", x(lastSlice.slice_start))
                .attr("cy", y(lastSlice.drift))
                .attr("r", 4)
                .attr("fill", "red");
        }
    });

    // ---------------- UI ----------------
    return (
        <div class={styles.distChartWrapper} style={{ width: "100%", height: "100%" }}>
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

            <svg
                ref={el => (ref = el)}
                style={{ width: "100%", height: "100%" }}
            />
        </div>
    );
}