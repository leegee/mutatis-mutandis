import { createEffect, createSignal, onMount, onCleanup } from "solid-js";
import * as d3 from "d3";
import type { SliceView } from "../types";
import styles from './DriftChart.module.css';

const MARGIN = {
    top: 10,
    right: 20,
    bottom: 30,
    left: 40
};

interface DriftChartProps {
    slice: SliceView;
    width?: number;
    height?: number;
    onSelectSlice?: (t: number) => void;
}

export default function DriftChart(props: DriftChartProps) {
    let ref: SVGSVGElement | undefined;

    // --- size tracking ---
    const [size, setSize] = createSignal({ width: 0, height: 0 });

    onMount(() => {
        if (!ref) return;

        const observer = new ResizeObserver(([entry]) => {
            const { width, height } = entry.contentRect;
            setSize({ width, height });
        });

        observer.observe(ref);
        onCleanup(() => observer.disconnect());
    });

    // reactive width/height
    const width = () => props.width ?? size().width;
    const height = () => props.height ?? size().height;

    createEffect(() => {
        const slice = props.slice;
        const w = width();
        const h = height();

        if (!slice || !ref || w === 0 || h === 0) return;

        const data = slice.history;

        const svg = d3.select(ref);
        svg.selectAll("*").remove();

        const xExtent = d3.extent(data, d => d.t) as [number, number];
        const yExtent = d3.extent(data, d => d.drift) as [number, number];

        const x = d3.scaleLinear()
            .domain(xExtent)
            .range([MARGIN.left, w - MARGIN.right]);

        const y = d3.scaleLinear()
            .domain(yExtent)
            .nice()
            .range([h - MARGIN.bottom, MARGIN.top]);

        const line = d3.line<typeof data[0]>()
            .x(d => x(d.t))
            .y(d => y(d.drift))
            .curve(d3.curveMonotoneX);

        svg.append("path")
            .attr('class', styles.driftPath)
            .datum(data)
            .attr("d", line);

        svg.selectAll("circle")
            .data(data)
            .enter()
            .append("circle")
            .attr('class', styles.driftPoint)
            .attr("cx", d => x(d.t))
            .attr("cy", d => y(d.drift))
            .attr("r", 6)
            .on("click", (_, d) => props.onSelectSlice?.(d.t));

        svg.selectAll("line.shock")
            .data(slice.transitions)
            .enter()
            .append("line")
            .attr('class', styles.driftShock)
            .attr("x1", t => x(t))
            .attr("x2", t => x(t))
            .attr("y1", MARGIN.top)
            .attr("y2", h - MARGIN.bottom);

        const current = data.find(d => d.t === slice.slice_start);

        if (current) {
            svg.append("circle")
                .attr("cx", x(current.t))
                .attr("cy", y(current.drift))
                .attr("r", 4)
                .attr("fill", "red");
        }
    });

    return (
        <svg
            ref={el => (ref = el)}
            style={{ width: "100%", height: "100%" }}
        />
    );
}
