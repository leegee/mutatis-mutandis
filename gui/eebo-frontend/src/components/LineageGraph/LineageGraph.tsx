import { createEffect, createSignal, onMount } from "solid-js";
import * as d3 from "d3";

import styles from "./LineageGraph.module.css";



type LineageNode = {
    id: string;
    year: number;
    cluster: number;
    size: number;
    lineage?: number;
    local?: {
        x: number;
        y: number;
    };

    global?: {
        x: number;
        y: number;
    };
};


type LineageLink = {
    source: string;
    target: string;
    similarity: number;
    confidence: number;
    type: string;
};

type LineageData = {
    concept: string;
    nodes: LineageNode[];
    links: LineageLink[];
};



export default function LineageGraph() {
    let svgRef!: SVGSVGElement;

    const [data, setData] = createSignal<LineageData>();

    onMount(async () => {
        const response = await fetch(
            "/lineage/CHURCH_lineage.json"
        );

        setData(await response.json());
    });

    createEffect(() => {
        const graph = data();

        if (!graph || !svgRef)
            return;

        render(graph);
    });

    function render(graph: LineageData) {
        const svg = d3.select(svgRef);
        svg.selectAll("*").remove();

        const rect = svgRef.getBoundingClientRect();
        const height = rect.height;

        const years = [
            ...new Set(
                graph.nodes.map(n => n.year)
            )
        ].sort();


        //
        // virtual timeline width
        //

        const yearSpacing = 180;

        const margin = {
            left: 120,
            right: 120,
            top: 80,
            bottom: 80
        };


        const graphWidth =
            margin.left +
            margin.right +
            Math.max(
                0,
                years.length - 1
            ) * yearSpacing;

        svg.attr(
            "viewBox", `0 0 ${ graphWidth } ${ height }`
        ).attr(
            "preserveAspectRatio",
            "xMinYMin meet"
        );

        const x = d3.scaleOrdinal<number, number>()
            .domain(years)
            .range(
                years.map(
                    (_, i) =>
                        margin.left + i * yearSpacing
                )
            );

        const y = d3.scaleLinear()
            .domain(
                d3.extent(
                    graph.nodes,
                    d => d.local!.y
                ) as [number, number]
            )
            .range([80, height - 80]);

        const positions = new Map<string, [number, number]>();

        for (const node of graph.nodes) {
            positions.set(node.id, [
                x(node.year)!,
                y(node.local?.y ?? 0)
            ]);
        }

        //
        // container
        //

        const root = svg.append("g");

        //
        // edges
        //

        root.selectAll("path")
            .data(graph.links)
            .enter()
            .append("path")
            .attr("class", styles.edge)
            .attr(
                "d",
                link => {
                    const a = positions.get(link.source);
                    const b = positions.get(link.target);

                    if (!a || !b)
                        return "";

                    const mid = (a[0] + b[0]) / 2;

                    return `
                M ${ a[0] } ${ a[1] }
                C ${ mid } ${ a[1] },
                  ${ mid } ${ b[1] },
                  ${ b[0] } ${ b[1] }
            `;
                }
            )
            .attr("stroke-width", d => Math.max(1, d.confidence * 8))
            .attr("opacity", d => Math.max(0.2, d.similarity))
            .attr("stroke-dasharray", d => d.type === "CONTINUATION" ? null : "6,4"
            );

        //
        // nodes
        //

        const radius = d3.scaleSqrt()
            .domain([
                0,
                d3.max(graph.nodes, d => d.size) ?? 1
            ])
            .range([6, 40]);

        const lineageColour =
            d3.scaleOrdinal<number, string>()
                .domain(
                    graph.nodes.map(n => n.lineage!)
                )
                .range(
                    d3.quantize(
                        d3.interpolateRainbow,
                        new Set(
                            graph.nodes.map(n => n.lineage)
                        ).size
                    )
                );

        root.selectAll("circle")
            .data(graph.nodes)
            .enter()
            .append("circle")
            .attr("class", styles.node)
            .attr("cx", d => positions.get(d.id)![0])
            .attr("cy", d => positions.get(d.id)![1])
            .attr("r", d => radius(d.size))
            // .attr("fill", d => yearColour(d.year))
            .attr("fill", d => lineageColour(d.lineage ?? 0))
            .append("title")
            .text(
                d =>
                    `${ graph.concept }
year ${ d.year }
cluster ${ d.cluster }
mass ${ d.size }`
            );

        //
        // years
        //

        root.selectAll("text")
            .data(years)
            .enter()
            .append("text")
            .attr("x", y => x(y)!)
            .attr("y", 40)
            .attr("text-anchor", "middle")
            .text(y => y);
    }

    return (
        <article class={styles.component}>
            <svg ref={svgRef} />
        </article>
    );
}
