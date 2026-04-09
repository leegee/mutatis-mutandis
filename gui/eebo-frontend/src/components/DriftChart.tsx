import { createEffect, onCleanup } from "solid-js";
import * as d3 from "d3";
import type { DriftChartProps, Neighbor } from "../types";

export default function DriftChart(props: DriftChartProps) {
    let svgRef: SVGSVGElement;

    createEffect(() => {
        const { data, hovered, setHovered, selected, setSelected } = props;
        if (!data) return;

        const width = 1000;
        const height = 300;
        const margin = { top: 20, right: 20, bottom: 30, left: 50 };

        const svg = d3.select(svgRef);
        svg.selectAll("*").remove();

        const tokens = Object.keys(data);
        const allSlices = tokens.flatMap(t => data[t].slices);
        const allYears = Array.from(new Set(allSlices.map(s => s.year))).sort((a, b) => a - b);

        const xScale = d3.scaleLinear()
            .domain([Math.min(...allYears), Math.max(...allYears)])
            .range([margin.left, width - margin.right]);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(allSlices.map(s => Math.max(s.drift, s.js_divergence))) || 1])
            .range([height - margin.bottom, margin.top]);

        // Lines
        tokens.forEach((token, idx) => {
            const slices = data[token].slices;
            const color = d3.schemeCategory10[idx % 10];

            const lineDrift = d3.line<any>()
                .x(d => xScale(d.year))
                .y(d => yScale(d.drift));

            svg.append("path")
                .datum(slices)
                .attr("fill", "none")
                .attr("stroke", color)
                .attr("stroke-width", 2)
                .attr("d", lineDrift)
                .on("mouseover", (_, d) => setHovered({ token, year: d[0].year, color }))
                .on("mouseout", () => setHovered(null))
                .on("click", (_, d) => setSelected({ token, year: d[0].year, color }));
        });

        // Axes
        svg.append("g")
            .attr("transform", `translate(0,${height - margin.bottom})`)
            .call(d3.axisBottom(xScale).tickFormat(d3.format("d")));

        svg.append("g")
            .attr("transform", `translate(${margin.left},0)`)
            .call(d3.axisLeft(yScale));

    });

    return <svg ref={svgRef} width={1000} height={300} />;
}
