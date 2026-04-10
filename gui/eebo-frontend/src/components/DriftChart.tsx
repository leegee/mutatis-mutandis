import { createEffect, onCleanup } from "solid-js";
import * as d3 from "d3";
import type { SliceView } from "../types";

type Props = {
    slice: SliceView | undefined;
};

export default function DriftChart(props: Props) {
    let ref: SVGSVGElement | undefined;

    createEffect(() => {
        const slice = props.slice;
        if (!slice || !ref) return;

        const data = slice.history;
        if (!data.length) return;

        const width = 600;
        const height = 200;

        const margin = {
            top: 10,
            right: 20,
            bottom: 30,
            left: 40
        };

        // clear
        const svg = d3.select(ref);
        svg.selectAll("*").remove();

        // scales
        const x = d3
            .scaleLinear()
            .domain(d3.extent(data, d => d.t) as [number, number])
            .range([margin.left, width - margin.right]);

        const y = d3
            .scaleLinear()
            .domain([0, d3.max(data, d => d.drift) || 1])
            .nice()
            .range([height - margin.bottom, margin.top]);

        // line (drift)
        const line = d3
            .line<typeof data[0]>()
            .x(d => x(d.t))
            .y(d => y(d.drift))
            .curve(d3.curveMonotoneX);

        svg
            .append("path")
            .datum(data)
            .attr("fill", "none")
            .attr("stroke", "black")
            .attr("stroke-width", 1.5)
            .attr("d", line);

        // 🔥 transitions (from d2 spikes)
        svg
            .selectAll(".shock")
            .data(slice.transitions ?? [])
            .enter()
            .append("line")
            .attr("class", "shock")
            .attr("x1", t => x(t))
            .attr("x2", t => x(t))
            .attr("y1", margin.top)
            .attr("y2", height - margin.bottom)
            .attr("stroke", "orange")
            .attr("stroke-dasharray", "4 2");

        // current slice marker
        const current = data.find(d => d.t === slice.slice_start);

        if (current) {
            svg
                .append("circle")
                .attr("cx", x(current.t))
                .attr("cy", y(current.drift))
                .attr("r", 4)
                .attr("fill", "red");
        }

        // axes
        const xAxis = d3.axisBottom(x).ticks(5).tickFormat(d3.format("d"));
        const yAxis = d3.axisLeft(y).ticks(4);

        svg
            .append("g")
            .attr("transform", `translate(0, ${height - margin.bottom})`)
            .call(xAxis);

        svg
            .append("g")
            .attr("transform", `translate(${margin.left}, 0)`)
            .call(yAxis);
    });

    return <svg ref={ref} width={600} height={200} />;
}
