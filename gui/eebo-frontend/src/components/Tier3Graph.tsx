import {
    createSignal,
    Show,
    createMemo,
    createEffect,
    onMount
} from "solid-js";

import * as d3 from "d3";

import type {
    Tier3GraphData,
    Tier3TokenGraph,
    Tier3Node,
    Tier3Link
} from "../types";


type Props = {
    data: any;
};


/*
Layout is purely deterministic scaffolding.

We only encode:
    - slice ordering (x axis)
    - within-slice cluster ordering (y axis)

No semantic geometry is assumed.
*/
function buildLayout(nodes: Tier3Node[]) {

    const slices = Array.from(
        new Set(nodes.map(n => n.slice))
    ).sort();

    const sliceIndex = new Map<string, number>();
    slices.forEach((s, i) => sliceIndex.set(s, i));

    const grouped = new Map<string, Tier3Node[]>();

    for (const n of nodes) {
        if (!grouped.has(n.slice)) grouped.set(n.slice, []);
        grouped.get(n.slice)!.push(n);
    }

    const yIndex = new Map<string, number>();

    for (const [slice, group] of grouped.entries()) {
        group.sort((a, b) => (b.size ?? 1) - (a.size ?? 1));

        group.forEach((node, idx) => {
            yIndex.set(`${node.slice}:${node.cluster}`, idx);
        });
    }

    return nodes.map(n => ({
        ...n,
        x0: (sliceIndex.get(n.slice) ?? 0) * 140 + 140,
        y0: (yIndex.get(`${n.slice}:${n.cluster}`) ?? 0) * 50 + 120,
        x: (sliceIndex.get(n.slice) ?? 0) * 140 + 140,
        y: (yIndex.get(`${n.slice}:${n.cluster}`) ?? 0) * 50 + 120
    }));
}


/*
Safely unwraps Tier2.7 export formats:

Expected:
    props.data[token].nodes
    props.data[token].links

*/
function unwrapGraph(data: any, token: string): Tier3TokenGraph | null {

    if (!data) return null;

    const root = data;
    const g = root[token];

    if (!g) return null;

    return {
        nodes: Array.isArray(g.nodes) ? g.nodes : [],
        links: Array.isArray(g.links) ? g.links : [],
        token
    };
}


export default function Tier3Graph(props: Props) {

    let svgRef: SVGSVGElement | undefined;

    const [selectedToken, setSelectedToken] =
        createSignal<string | null>(null);

    const [selectedNode, setSelectedNode] =
        createSignal<Tier3Node | null>(null);

    const tokenKeys = () => {
        const root = props.data ?? {};
        return Object.keys(root);
    };


    const graph = createMemo<Tier3TokenGraph | null>(() => {

        const keys = tokenKeys();
        const token = selectedToken() ?? keys[0];

        if (!token) return null;

        return unwrapGraph(props.data, token);
    });


    onMount(() => {
        const width = 1000;
        const height = 600;

        const svg = d3.select(svgRef!)
            .attr("viewBox", [0, 0, width, height] as any)
            .style("width", "100%")
            .style("height", "50vh");

        const container = svg.append("g");

        svg.call(
            d3.zoom<SVGSVGElement, unknown>()
                .scaleExtent([0.25, 5])
                .on("zoom", (event) => {
                    container.attr("transform", event.transform);
                })
        );

        const simulation =
            d3.forceSimulation<Tier3Node>()

                .force(
                    "x",
                    d3.forceX<Tier3Node>(d => (d as any).x0)
                        .strength(0.85)
                )
                .force(
                    "y",
                    d3.forceY<Tier3Node>(d => (d as any).y0)
                        .strength(0.35)
                )
                .force(
                    "link",
                    d3.forceLink<Tier3Node, any>()
                        .id((d: any) => d.id)
                        .distance(50)
                        .strength(0.08)
                )
                .force(
                    "collide",
                    d3.forceCollide<Tier3Node>()
                        .radius(d => Math.max(10, Math.sqrt(d.size || 1) * 2))
                        .strength(0.9)
                )
                .force(
                    "charge",
                    d3.forceManyBody().strength(-25)
                )
                .alphaDecay(0.04)
                .velocityDecay(0.5);


        let linkSel: any;
        let nodeSel: any;

        const render = () => {
            const g = graph();
            if (!g) return;

            const nodes = buildLayout(g.nodes ?? []);
            const links = g.links ?? [];

            // CRITICAL: update simulation state only
            simulation.nodes(nodes as any);
            (simulation.force("link") as any).links(links);

            // DO NOT delete everything
            // container.selectAll("*").remove(); ❌ REMOVE THIS

            // ---- LINKS JOIN ----
            linkSel = container
                .selectAll("line")
                .data(links, (d: any) => `${d.source}->${d.target}`)
                .join(
                    enter => enter.append("line")
                        .attr("stroke", "#bbb")
                        .attr("stroke-width", 6)
                        .attr("stroke-opacity", 0.3),
                    update => update,
                    exit => exit.remove()
                );

            // ---- NODES JOIN ----
            nodeSel = container
                .selectAll("circle")
                .data(nodes, (d: any) => d.id)
                .join(
                    enter => enter.append("circle")
                        .attr("fill", "#4a90e2")
                        .style("cursor", "pointer")
                        .on("click", (_: any, d: Tier3Node) => {
                            setSelectedNode(d);
                        }),
                    update => update,
                    exit => exit.remove()
                )
                .attr("r", d => Math.max(6, Math.sqrt(d.size || 1) * 2));

            simulation.alpha(1).restart();
        };

        simulation.on("tick", () => {

            if (!linkSel || !nodeSel) return;

            linkSel
                .attr("x1", (d: any) => d.source.x)
                .attr("y1", (d: any) => d.source.y)
                .attr("x2", (d: any) => d.target.x)
                .attr("y2", (d: any) => d.target.y);

            nodeSel
                .attr("cx", (d: any) => d.x)
                .attr("cy", (d: any) => d.y);
        });


        createEffect(() => {
            graph();
            render();
        });

    });


    return (
        <article class="responsive">

            <section class="no-margin border">
                <aside class="padding border">
                    <label>Token</label>
                    <select
                        value={selectedToken() ?? ""}
                        onChange={(e) =>
                            setSelectedToken(e.currentTarget.value)
                        }
                    >
                        {tokenKeys().map(token => (
                            <option value={token}>{token}</option>
                        ))}
                    </select>
                </aside>

                <svg class="" ref={svgRef}></svg>
            </section>

            <section class="padding border">
                <h3>Node Inspector</h3>

                <Show when={selectedNode()}>
                    {(n) => (
                        <div>
                            <p><b>ID:</b> {n().id}</p>
                            <p><b>Slice:</b> {n().slice}</p>
                            <p><b>Cluster:</b> {n().cluster}</p>
                            <p><b>Size:</b> {n().size ?? 1}</p>

                            <hr />

                            <p>Nodes are structural states.</p>
                            <p>Links encode continuity strength.</p>
                        </div>
                    )}
                </Show>
            </section>

        </article>
    );
}