import { createEffect } from "solid-js";
import * as d3 from "d3";
import type { SliceView } from "../types";

type GraphNode = d3.SimulationNodeDatum & {
    id: string;
    similarity: number;
    rank: number;
};

type GraphLink = d3.SimulationLinkDatum<GraphNode> & {
    weight: number;
};

type Props = {
    slice: SliceView;
};

export default function NeighborGraph(props: Props) {
    let svgRef: SVGSVGElement | undefined;

    // re-run when slice changes
    createEffect(() => {
        const slice = props.slice;
        if (!slice || !svgRef) return;

        console.log("[NeighborGraph] render", {
            token: slice.token,
            neighbors: slice.neighbors.length,
            drift: slice.drift
        });

        const width = 800;
        const height = 600;

        const svg = d3.select(svgRef);
        svg.selectAll("*").remove(); // clear previous render

        // --- nodes ---
        const nodes: GraphNode[] = slice.neighbors.map((n) => ({
            id: n.token,
            similarity: n.similarity,
            rank: slice.rank.get(n.token) ?? 0
        }));

        // include central token
        nodes.push({
            id: slice.token,
            similarity: 1,
            rank: -1
        });

        // --- links ---
        const links: GraphLink[] = slice.neighbors.map((n) => ({
            source: slice.token,
            target: n.token,
            weight: 1 - n.similarity
        }));

        // --- simulation ---
        const simulation = d3
            .forceSimulation<GraphNode>(nodes)
            .force(
                "link",
                d3
                    .forceLink<GraphNode, GraphLink>(links)
                    .id((d) => d.id)
                    .distance((d) => 60 + d.weight * 180)
            )
            .force(
                "charge",
                d3.forceManyBody<GraphNode>().strength((d) =>
                    d.id === slice.token ? -300 : -120
                )
            )
            .force("center", d3.forceCenter(width / 2, height / 2));

        // --- links render ---
        const link = svg
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("stroke", "#999")
            .attr("stroke-opacity", 0.6);

        // --- nodes render ---
        const node = svg
            .selectAll("circle")
            .data(nodes)
            .join("circle")
            .attr("r", (d) => (d.id === slice.token ? 10 : 6))
            .attr("fill", (d) => {
                if (d.id === slice.token) return "#000";

                // drift-aware coloring
                return d3.interpolateTurbo(slice.normalizedDrift);
            });

        // --- labels ---
        const labels = svg
            .selectAll("text")
            .data(nodes)
            .join("text")
            .text((d) => d.id)
            .attr("font-size", 10)
            .attr("dx", 8)
            .attr("dy", "0.35em");

        // --- drag (fixed typing) ---
        const drag = d3
            .drag<Element, GraphNode>()
            .on("start", (event, d) => {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on("drag", (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on("end", (event, d) => {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });

        node.call(drag as any);

        // --- tick ---
        simulation.on("tick", () => {
            link
                .attr("x1", (d: any) => d.source.x ?? 0)
                .attr("y1", (d: any) => d.source.y ?? 0)
                .attr("x2", (d: any) => d.target.x ?? 0)
                .attr("y2", (d: any) => d.target.y ?? 0);

            node.attr("cx", (d) => d.x ?? 0).attr("cy", (d) => d.y ?? 0);

            labels
                .attr("x", (d) => d.x ?? 0)
                .attr("y", (d) => d.y ?? 0);
        });
    });

    return <svg ref={(el) => (svgRef = el)} width={800} height={600} />;
}
