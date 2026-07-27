import { createEffect, createSignal, onMount, onCleanup, Show } from "solid-js";
import * as d3 from "d3";

import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "../../corpus_config";

import styles from "./LineageGraph.module.css";
import Tooltip from "./Tooltip";
import DetailPanel from "./DetailPanel";
import type { LineageData, LineageNode, ViewportRatio, ScrollState, } from "./types";

const YEAR_RANGE_FROM_DATA = false;

type LineageGraphProps = {
    data: LineageData;

    // "detail" (default): the full, scrollable, interactive timeline.
    // "overview": a compact, full-width strip with no scroll -- every
    // year fitted into the available width -- used as a minimap.
    variant?: "detail" | "overview";

    // --- overview mode only ---
    // Range (as 0..1 fractions of the detail view's scroll width) to
    // highlight as "currently visible in the paired detail view".
    viewport?: ViewportRatio;
    // Called with a 0..1 position when the overview strip is clicked
    // or dragged, so the parent can scroll the detail view to match.
    onNavigate?: (ratio: number) => void;

    // --- detail mode only ---
    // Hands the detail view's scrollable DOM node up to the parent,
    // so the parent can imperatively scroll it in response to
    // onNavigate from the paired overview.
    onContainerReady?: (el: HTMLDivElement) => void;
    // Reports the detail view's scroll position whenever it changes,
    // so the parent can compute the overview's highlighted band.
    onViewportChange?: (state: ScrollState) => void;
};

type TooltipState = {
    node: LineageNode;
    x: number;
    y: number;
};

/**
 * Projects a set of 2D points onto their principal axis (first principal
 * component), returning one scalar per point.
 *
 * Why: a UMAP embedding's two output axes carry no inherent ranking or
 * meaning -- there's no reason "y" holds more real structure than "x".
 * The principal axis is, provably (Eckart-Young), the single linear
 * projection that preserves the most variance of any 2D-to-1D linear
 * projection, so we compute the direction actually justified by the
 * data's own spread rather than picking an axis arbitrarily.
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

    const trace = varX + varY;
    const det = varX * varY - covXY * covXY;
    const discriminant = Math.sqrt(Math.max(0, trace * trace - 4 * det));
    const lambdaMax = (trace + discriminant) / 2;

    let ex = covXY;
    let ey = lambdaMax - varX;

    const norm = Math.hypot(ex, ey);

    if (norm < 1e-9) {
        ex = 1;
        ey = 0;
    } else {
        ex /= norm;
        ey /= norm;
    }

    return points.map(p => (p.x - meanX) * ex + (p.y - meanY) * ey);
}



export default function LineageGraph(props: LineageGraphProps) {
    let svgRef!: SVGSVGElement;
    let containerRef!: HTMLDivElement;

    const [tooltip, setTooltip] = createSignal<TooltipState | null>(null);
    const [selectedNode, setSelectedNode] = createSignal<LineageNode | null>(null);

    const variant = () => props.variant ?? "detail";

    onMount(() => {
        if (variant() !== "detail")
            return;

        props.onContainerReady?.(containerRef);

        const reportScroll = () => {
            props.onViewportChange?.({
                scrollLeft: containerRef.scrollLeft,
                scrollWidth: containerRef.scrollWidth,
                clientWidth: containerRef.clientWidth,
            });
        };

        const onWheel = (e: WheelEvent) => {
            // Ignore pinch zoom
            if (e.ctrlKey)
                return;

            // If the user is already scrolling horizontally (ie trackpad), preserve that.
            const delta = Math.abs(e.deltaX) > Math.abs(e.deltaY)
                ? e.deltaX
                : e.deltaY;

            if (delta !== 0) {
                e.preventDefault();
                containerRef.scrollLeft += delta;
            }
        };

        containerRef.addEventListener("wheel", onWheel, { passive: false });
        containerRef.addEventListener("scroll", reportScroll, { passive: true });

        // Layout (and therefore scrollWidth) isn't settled on the very
        // first tick -- report once after the browser's had a frame.
        requestAnimationFrame(reportScroll);

        onCleanup(() => containerRef.removeEventListener("scroll", reportScroll));
    });

    createEffect(() => {
        if (!svgRef || !props.data)
            return;

        render(props.data);
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

        const isOverview = variant() === "overview";

        const rect = svgRef.getBoundingClientRect();
        const height = rect.height;

        const years = [
            ...new Set(
                YEAR_RANGE_FROM_DATA
                    ? graph.nodes.map(n => n.year)
                    : d3.range(CORPUS_START_YEAR, CORPUS_END_YEAR + 1)
            )
        ].sort();

        const margin = isOverview
            ? { left: 0, right: 10, top: 4, bottom: 4 }
            : { left: 120, right: 120, top: 80, bottom: 80 };

        let x: d3.ScaleOrdinal<number, number>;
        let graphWidth: number;

        if (isOverview) {
            // Overview fits width
            const containerRect = containerRef.getBoundingClientRect();
            graphWidth = Math.max(containerRect.width, 1);

            const usableWidth = graphWidth - margin.left - margin.right;

            x = d3.scaleOrdinal<number, number>()
                .domain(years)
                .range(
                    years.map((_, i) =>
                        years.length <= 1
                            ? graphWidth / 2
                            : margin.left + (i / (years.length - 1)) * usableWidth
                    )
                );
        } else {
            const yearSpacing = 180;

            graphWidth =
                margin.left + margin.right +
                Math.max(0, years.length - 1) * yearSpacing;

            x = d3.scaleOrdinal<number, number>()
                .domain(years)
                .range(
                    years.map((_, i) => margin.left + i * yearSpacing)
                );
        }

        svg.attr("viewBox", `0 0 ${ graphWidth } ${ height }`)
            .attr("preserveAspectRatio", "xMinYMin meet");

        if (!isOverview) {
            // Clicking empty canvas dismisses the detail panel.
            svg.on("click", () => setSelectedNode(null));
        }

        const principalCoords = projectOntoPrincipalAxis(
            graph.nodes.map(n => ({
                x: n.global?.x ?? 0,
                y: n.global?.y ?? 0,
            }))
        );

        const y = d3.scaleLinear()
            .domain(d3.extent(principalCoords) as [number, number])
            .range([margin.top, height - margin.bottom]);

        const positions = new Map<string, [number, number]>();

        graph.nodes.forEach((node, i) => {
            positions.set(node.id, [
                x(node.year)!,
                y(principalCoords[i])
            ]);
        });

        const radius = d3.scaleSqrt()
            .domain([
                0,
                d3.max(graph.nodes, d => d.size) ?? 1
            ])
            .range(isOverview ? [1, 5] : [6, 40]);

        if (!isOverview) {
            // The PCA axis preserves true relative distance -- including
            // when several nodes are genuinely almost co-located (e.g. an
            // early, narrow year whose whole spread of clusters occupies
            // a tiny corner of the corpus-wide embedding). That's correct
            // data, but two circles can't render on top of each other and
            // both stay legible. This nudges only nodes that actually
            // overlap at render scale apart, pulled back toward their
            // true position the rest of the time -- it adds no
            // information, it just stops real proximity from becoming
            // total occlusion. Column spacing (yearSpacing) already
            // exceeds twice the max node radius, so this naturally stays
            // scoped within each year without needing to constrain it
            // explicitly.
            const simNodes = graph.nodes.map(node => {
                const [px, py] = positions.get(node.id)!;
                return { id: node.id, x: px, y: py, targetX: px, targetY: py };
            });

            d3.forceSimulation(simNodes)
                .force("x", d3.forceX<typeof simNodes[number]>(d => d.targetX).strength(0.9))
                .force("y", d3.forceY<typeof simNodes[number]>(d => d.targetY).strength(0.9))
                .force(
                    "collide",
                    d3.forceCollide<typeof simNodes[number]>(
                        d => radius(graph.nodes.find(n => n.id === d.id)!.size) + 1
                    )
                )
                .stop()
                .tick(120);

            for (const sim of simNodes) {
                positions.set(sim.id, [sim.x, sim.y]);
            }
        }

        const root = svg.append("g");

        if (isOverview) {
            const vp = props.viewport ?? { startRatio: 0, endRatio: 1 };
            const usableWidth = graphWidth - margin.left - margin.right;

            // Highlights which slice of the (much wider) detail view is
            // currently scrolled into view.
            root.append("rect")
                .attr("class", styles.viewportBand)
                .attr("x", margin.left + vp.startRatio * usableWidth)
                .attr("y", 0)
                .attr(
                    "width",
                    Math.max(2, (vp.endRatio - vp.startRatio) * usableWidth)
                )
                .attr("height", height);

            const navigate = (event: MouseEvent) => {
                const localX = pointerPosition(event).x;
                const ratio = (localX - margin.left) / usableWidth;
                props.onNavigate?.(Math.max(0, Math.min(1, ratio)));
            };

            let dragging = false;

            svg.on("mousedown", (event: MouseEvent) => {
                dragging = true;
                navigate(event);
            });

            svg.on("mousemove", (event: MouseEvent) => {
                if (dragging)
                    navigate(event);
            });

            const stopDrag = () => { dragging = false; };

            svg.on("mouseup", stopDrag);
            svg.on("mouseleave", stopDrag);
        }

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
                d => Math.max(isOverview ? 0.5 : 1, d.confidence * (isOverview ? 3 : 8))
            )
            .attr("opacity", d => Math.max(0.2, d.similarity))
            .attr(
                "stroke-dasharray",
                d => (!isOverview && d.type !== "CONTINUATION") ? "6,4" : null
            );

        //
        // nodes
        //
        // (radius scale computed earlier, alongside the collision pass)

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

        const circles = root.selectAll("circle")
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
            );

        if (!isOverview) {
            // Per-node hover/click detail only makes sense at detail
            // scale -- at a few em tall, individual nodes are navigation
            // targets for the strip as a whole, not inspectable on their
            // own.
            circles
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
                    if (selectedNode()?.id === d.id) {
                        event.target.classList.remove(styles.nodeSelected);
                        setSelectedNode(null);
                    } else {
                        event.target.classList.add(styles.nodeSelected);
                        setSelectedNode(d);
                    }
                });

            //
            // years
            //

            root.selectAll("text")
                .data(years)
                .enter()
                .append("text")
                .attr("x", yr => x(yr)!)
                .attr("y", 40)
                .attr("text-anchor", "middle")
                .text(yr => yr);
        }
    }

    return (
        <article
            class={
                variant() === "overview"
                    ? `${ styles.component } ${ styles.overview }`
                    : styles.component
            }
            ref={containerRef}
        >
            <svg ref={svgRef} />

            <Show when={variant() === "detail" && tooltip()}>
                {t => <Tooltip tooltip={t()} concept={props.data.concept} />}
            </Show>

            <Show when={variant() === "detail" && selectedNode()}>
                {n => (
                    <DetailPanel
                        node={n()}
                        concept={props.data.concept}
                        onClose={() => setSelectedNode(null)}
                    />
                )}
            </Show>
        </article>
    );
}
