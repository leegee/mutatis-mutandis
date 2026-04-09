import { createEffect, onCleanup } from "solid-js";
import * as d3 from "d3";
import type { NeighborGraphProps } from "./types";

export default function NeighborGraph(props: NeighborGraphProps) {
    let svgRef: SVGSVGElement;

    createEffect(() => {
        const { token, neighbors, drift, color } = props;
        if (!token || !neighbors?.length) return;

        const width = 800;
        const height = 600;
        const svg = d3.select(svgRef);
        svg.selectAll("*").remove();

        const nodes = [
            { id: token, central: true, sim: -1, count: -1 },
            ...neighbors.map(n => ({ id: n.token, central: false, sim: n.similarity, count: n.count }))
        ];

        const links = neighbors.map(n => ({ source: token, target: n.token, weight: 1 - n.similarity }));

        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(d => 80 + d.weight * 150))
            .force("charge", d3.forceManyBody().strength(d => d.central ? -300 : -120 * Math.log1p(d.count)))
            .force("center", d3.forceCenter(width / 2, height / 2));

        // Edges
        const link = svg.append("g")
            .selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("stroke", "#aaa");

        // Nodes
        const node = svg.append("g")
            .selectAll("circle")
            .data(nodes)
            .enter().append("circle")
            .attr("r", d => d.central ? 20 : 5 + 8 * d.sim + Math.log1p(d.count) * 3)
            .attr("fill", color)
            .attr("stroke", "white")
            .attr("stroke-width", 1.5)
            .call(d3.drag()
                .on("start", dragstart)
                .on("drag", dragged)
                .on("end", dragend)
            );

        // Labels
        const text = svg.append("g")
            .selectAll("text")
            .data(nodes)
            .enter().append("text")
            .text(d => d.id)
            .attr("font-size", d => d.central ? 24 : 14)
            .attr("fill", "white")
            .attr("stroke", "black")
            .attr("stroke-width", 3)
            .attr("paint-order", "stroke");

        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x).attr("y2", d => d.target.y);

            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);

            text
                .attr("x", d => d.x)
                .attr("y", d => d.y - 10);
        });

        function dragstart(event, d) { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }
        function dragged(event, d) { d.fx = event.x; d.fy = event.y; }
        function dragend(event, d) { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }

        onCleanup(() => simulation.stop());

    });

    return <svg ref={svgRef} width={800} height={600} />;
}
