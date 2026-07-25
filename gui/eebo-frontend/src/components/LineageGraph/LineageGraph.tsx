import { createEffect, createSignal, onMount } from "solid-js";
import { Show } from "solid-js";
import * as d3 from "d3";

import styles from "./LineageGraph.module.css";
import Tooltip from "./Tooltip";
import DetailPanel from "./DetailPanel";
import type { LineageData, TooltipState, LineageNode } from "./types";


/**
 * Projects a set of 2D points onto their principal axis (first principal
 * component), returning one scalar per point.
 *
 * Why: a UMAP embedding's two output axes carry no inherent ranking or
 * meaning -- there's no reason "y" holds more real structure than "x".
 * Rendering position from raw y alone (or from y plus an arbitrary x
 * jitter) risks flattening away separation that actually falls along
 * some other direction, e.g. a diagonal -- which is exactly what caused
 * genuinely distinct clusters to render on top of each other before.
 *
 * The principal axis is, provably (Eckart-Young), the single linear
 * projection that preserves the most variance -- equivalently, distorts
 * relative distances the least -- of any 2D-to-1D linear projection. So
 * rather than picking an axis, we compute the one direction that's
 * actually justified by the data's own spread.
 */
function projectOntoPrincipalAxis(
    points: { x: number; y: number }[]
): number[] {
    const n = points.length;

    if (n === 0)
        return [];

    const meanX = d3.mean(points, p => p.x) ?? 0;
    const meanY = d3.mean(points, p => p.y) ?? 0;

    let varX = 0, varY = 0, covXY = 0;

    for (const p of points) {
        const dx = p.x - meanX;
        const dy = p.y - meanY;
        varX += dx * dx;
        varY += dy * dy;
        covXY += dx * dy;
    }

    varX /= n;
    varY /= n;
    covXY /= n;

    // Closed-form largest eigenvalue of a symmetric 2x2 covariance matrix.
    const trace = varX + varY;
    const det = varX * varY - covXY * covXY;
    const discriminant = Math.sqrt(Math.max(0, trace * trace - 4 * det));
    const lambdaMax = (trace + discriminant) / 2;

    // Corresponding eigenvector = the principal axis direction.
    let ex = covXY;
    let ey = lambdaMax - varX;

    const norm = Math.hypot(ex, ey);

    if (norm < 1e-9) {
        // Degenerate (near-zero variance, or already axis-aligned):
        // fall back to the x-axis rather than divide by ~0.
        ex = 1;
        ey = 0;
    } else {
        ex /= norm;
        ey /= norm;
    }

    return points.map(p => (p.x - meanX) * ex + (p.y - meanY) * ey);
}



export default function LineageGraph() {
    let svgRef!: SVGSVGElement;
    let containerRef!: HTMLDivElement;

    const [data, setData] = createSignal<LineageData>();
    const [tooltip, setTooltip] = createSignal<TooltipState | null>(null);
    const [selectedNode, setSelectedNode] = createSignal<LineageNode | null>(null);

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

    function pointerPosition(event: MouseEvent) {
        const containerRect = containerRef.getBoundingClientRect();

        return {
            x: event.clientX - containerRect.left,
            y: event.clientY - containerRect.top,
        };
    }

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

        // Clicking empty canvas dismisses the detail panel.
        svg.on("click", () => setSelectedNode(null));

        const x = d3.scaleOrdinal<number, number>()
            .domain(years)
            .range(
                years.map(
                    (_, i) =>
                        margin.left + i * yearSpacing
                )
            );

        // Combine global.x and global.y into a single, principled vertical
        // coordinate rather than using global.y alone.
        const principalCoords = projectOntoPrincipalAxis(
            graph.nodes.map(n => ({
                x: n.global?.x ?? 0,
                y: n.global?.y ?? 0,
            }))
        );

        const y = d3.scaleLinear()
            .domain(d3.extent(principalCoords) as [number, number])
            .range([80, height - 80]);

        const positions = new Map<string, [number, number]>();

        graph.nodes.forEach((node, i) => {
            positions.set(node.id, [
                x(node.year)!,
                y(principalCoords[i])
            ]);
        });

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
            .attr("class", d =>
                d.id === selectedNode()?.id
                    ? `${ styles.node } ${ styles.nodeSelected }`
                    : styles.node
            )
            .attr("cx", d => positions.get(d.id)![0])
            .attr("cy", d => positions.get(d.id)![1])
            .attr("r", d => radius(d.size))
            .attr("fill", d => lineageColour(d.lineage ?? 0))
            .attr(
                "stroke-dasharray",
                d => d.lineage_stable === false ? "3,2" : null
            )
            .on("mouseenter", (event, d) => {
                const pos = pointerPosition(event);
                setTooltip({ node: d, x: pos.x, y: pos.y });
            })
            .on("mousemove", (event, d) => {
                const pos = pointerPosition(event);
                setTooltip({ node: d, x: pos.x, y: pos.y });
            })
            .on("mouseleave", () => setTooltip(null))
            .on("click", (event, d) => {
                event.stopPropagation();
                setTooltip(null);
                setSelectedNode(
                    selectedNode()?.id === d.id ? null : d
                );
            });

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
        <article class={styles.component} ref={containerRef}>
            <svg ref={svgRef} />

            <Show when={tooltip()}>
                {t => <Tooltip tooltip={t()} concept={data()?.concept} />}
            </Show>

            <Show when={selectedNode()}>
                {n => (
                    <DetailPanel
                        node={n()}
                        concept={data()?.concept}
                        onClose={() => setSelectedNode(null)}
                    />
                )}
            </Show>
        </article>
    );
}
