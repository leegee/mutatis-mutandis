import { createEffect, createSignal, onMount } from "solid-js";
import * as d3 from "d3";

import styles from "./LineageGraph.module.css";


type LineageNode = {
    id: string;
    year: number;
    cluster: number;
    size: number;

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



        //
        // nodes grouped by year
        //

        const nodesByYear =
            d3.group(
                graph.nodes,
                d => d.year
            );


        const positions = new Map<string, [number, number]>();

        for (const [year, nodes] of nodesByYear) {

            const totalHeight = nodes.length * 80;
            const start = (height - totalHeight) / 2;

            nodes.forEach(
                (node, index) => {
                    positions.set(
                        node.id,
                        [
                            x(year)!,
                            start + index * 80
                        ]
                    );
                }
            );
        }



        //
        // zoom container
        //

        const root = svg.append("g");

        d3.select(svgRef)
            .call(
                d3.zoom<SVGSVGElement, unknown>()
                    .scaleExtent([0.3, 4])
                    .on("zoom",
                        e => { root.attr("transform", e.transform); }
                    )
            );


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
            .attr(
                "stroke-width",
                d => Math.max(1, d.similarity * 5)
            );


        //
        // nodes
        //
        const radius = d3.scaleSqrt()
            .domain([
                0,
                d3.max(
                    graph.nodes,
                    d => d.size
                ) ?? 1
            ])
            .range([5, 35]);

        root.selectAll("circle")
            .data(graph.nodes)
            .enter()
            .append("circle")
            .attr("class", styles.node)
            .attr("cx", d => positions.get(d.id)![0])
            .attr("cy", d => positions.get(d.id)![1])
            .attr("r", d => radius(d.size))
            .append("title")
            .text(d =>
                `${ graph.concept }
${ d.year }
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
