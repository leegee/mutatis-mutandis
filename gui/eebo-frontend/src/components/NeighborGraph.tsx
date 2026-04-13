import { type Component, createMemo, For, onCleanup, onMount } from "solid-js";
import type { EventNeighbourhoodOpen, SliceView } from "../types";
import { eeboStore, setEeboStore, closeOverlay, toggleOverlay } from "../stores/Eebo.store";
import styles from "./NeighborGraph.module.css";

type Neighbor = SliceView["neighbors"][number];

const PADDING = 24;

export type NeighborGraphProps = {
    slice: SliceView;
    width: number;
    height: number;
    onClick: (e: MouseEvent) => void;
};

type PositionedNeighbor = Neighbor & {
    x: number;
    y: number;
};


function radialLayout(
    cx: number,
    cy: number,
    neighbors: Neighbor[]
): PositionedNeighbor[] {

    const PAD = 24;

    const maxRadius = Math.min(cx, cy) - PAD;
    const minRadius = maxRadius * 0.25;

    const sorted = [...neighbors].sort(
        (a, b) => (a.similarity ?? 0) - (b.similarity ?? 0)
    );

    const sims = sorted.map(d => d.similarity ?? 0);

    const q = (v: number) => {
        let i = 0;
        while (i < sims.length && sims[i] < v) i++;
        return sims.length <= 1 ? 1 : i / (sims.length - 1);
    };

    const totalWeight = sorted.reduce(
        (acc, n) => acc + Math.log1p(n.count ?? 1),
        0
    );

    let angleCursor = 0;

    return sorted.map((d) => {
        const sim = d.similarity ?? 0;
        const nq = q(sim);

        const radius =
            minRadius +
            (1 - nq) * (maxRadius - minRadius);

        const weight = Math.log1p(d.count ?? 1);
        const angleSpan = (weight / totalWeight) * 2 * Math.PI;

        angleCursor += angleSpan / 2;
        const angle = angleCursor;
        angleCursor += angleSpan / 2;

        return {
            ...d,
            x: cx + Math.cos(angle) * radius,
            y: cy + Math.sin(angle) * radius
        };
    });
}


function nodeSize(n: Neighbor): number {
    const sim = n.similarity ?? 0;
    const count = n.count ?? 1;

    return (
        8 +
        (1 - sim) * 14 +
        Math.log1p(count) * 3
    );
}


const NeighborGraph: Component<NeighborGraphProps> = (props) => {
    const neighbors = createMemo(() => props.slice.neighbors ?? []);

    const center = createMemo(() => ({
        x: props.width / 2,
        y: props.height / 2
    }));

    const sorted = createMemo(() =>
        [...neighbors()].sort(
            (a, b) => (b.similarity ?? 0) - (a.similarity ?? 0)
        )
    );

    const layout = createMemo<PositionedNeighbor[]>(() => {
        const c = center();
        return radialLayout(c.x, c.y, sorted());
    });

    // overlay event bridge
    onMount(() => {
        const handler = (e: Event) => {
            const ev = e as CustomEvent<EventNeighbourhoodOpen>;
            const d = ev.detail;

            if (!d?.token || d.slice_start == null) return;

            setEeboStore("selected", {
                token: d.token,
                slice_start: d.slice_start,
                slice_end: d.slice_end ?? d.slice_start,
                color: d.color ?? "red"
            });

            toggleOverlay(d.x, d.y);
        };

        window.addEventListener("neighbourhood:open", handler);

        onCleanup(() => {
            window.removeEventListener("neighbourhood:open", handler);
        });
    });

    onMount(() => {
        const handler = (e: MouseEvent) => {
            if (!eeboStore._overlay.open) return;

            const target = e.target as Node | null;
            const svgEl = document.querySelector("svg");

            if (svgEl && target && svgEl.contains(target)) {
                return;
            }

            closeOverlay();
        };

        document.addEventListener("mousedown", handler, true);

        onCleanup(() => document.removeEventListener("mousedown", handler, true));
    });

    return (
        <article class='surface'>
            <svg width={props.width} height={props.height}>
                {/* center node */}
                <g>
                    <circle
                        cx={center().x}
                        cy={center().y}
                        r={16}
                        fill={eeboStore.selected.color ?? "red"}
                    />

                    <text x={center().x} y={center().y} dy={4} text-anchor="middle">
                        {props.slice.token}
                    </text>

                    <text x={center().x} y={center().y} dy={26} text-anchor="middle">
                        {props.slice.slice_start} - {props.slice.slice_end}
                    </text>
                </g>

                {/* neighbors */}
                <For each={layout()}>
                    {(n) => (
                        <g>
                            <circle
                                cx={n.x}
                                cy={n.y}
                                r={nodeSize(n)}
                                fill={eeboStore.selected.color ?? "red"}
                                opacity={0.85}
                            />

                            <text x={n.x} y={n.y} dy={4} text-anchor="middle">
                                {n.token}
                            </text>

                            <text x={n.x} y={n.y} dy={20} text-anchor="middle">
                                {n.count ?? 0}
                            </text>

                            <text x={n.x} y={n.y} dy={36} text-anchor="middle">
                                {(n.similarity ?? 0).toFixed(4)}
                            </text>
                        </g>
                    )}
                </For>
            </svg>
        </article>
    );
};

export default NeighborGraph;
