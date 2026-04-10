import {
    type Component,
    createMemo,
    For,
    onMount,
    onCleanup
} from "solid-js";

import type {
    EventNeighbourhoodOpen,
    SliceView
} from "../types";

import styles from "./NeighborGraph.module.css";
import { eeboStore, setEeboStore } from "../stores/Eebo.store";

type Neighbor = SliceView["neighbors"][number];

export type NeighborGraphProps = {
    slice: SliceView;
    width: number;
    height: number;
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
    const n = neighbors.length || 1;

    const maxRadius = Math.min(cx, cy) * 0.9;
    const minRadius = maxRadius * 0.3;

    return neighbors.map((d, i) => {
        const sim = d.similarity ?? 0;

        const simScaled = (sim - 0.6) / 0.4;
        const clamped = Math.max(0, Math.min(1, simScaled));

        const spread = 1 - Math.pow(clamped, 2);

        const countWeight = Math.log1p(d.count ?? 1) / 4;

        const radius =
            minRadius +
            (maxRadius - minRadius) *
            spread *
            (1 + countWeight);

        const angle =
            (i / n) * 2 * Math.PI +
            ((d.count ?? 0) % 7) * 0.08;

        return {
            ...d,
            x: cx + Math.cos(angle) * radius,
            y: cy + Math.sin(angle) * radius
        };
    });
}

function nodeSize(n: Neighbor): number {
    return (
        10 +
        18 * (1 - (n.similarity ?? 0)) +
        Math.log1p(n.count ?? 1) * 4
    );
}

const NeighborGraph: Component<NeighborGraphProps> = (props) => {
    const neighbors = createMemo(() => {
        return props.slice.neighbors ?? [];
    });

    const layout = createMemo<PositionedNeighbor[]>(() => {
        const cx = props.width / 2;
        const cy = props.height / 2;

        const sorted = [...neighbors()].sort(
            (a, b) =>
                (b.similarity ?? 0) -
                (a.similarity ?? 0)
        );

        return radialLayout(cx, cy, sorted);
    });

    const center = createMemo(() => ({
        x: props.width / 2,
        y: props.height / 2
    }));

    onMount(() => {
        const handler = (e: Event) => {
            const ev =
                e as CustomEvent<EventNeighbourhoodOpen>;

            const {
                token,
                slice_start,
                slice_end,
                color,
                x,
                y
            } = ev.detail;

            if (!token || slice_start == null) return;

            // selected state
            setEeboStore("selected", {
                token,
                slice_start,
                slice_end: slice_end ?? slice_start,
                color
            });

            // overlay state (toggle if same point)
            setEeboStore("overlay", (prev) => ({
                open:
                    !prev.open ||
                    prev.x !== x ||
                    prev.y !== y,
                x,
                y
            }));
        };

        window.addEventListener(
            "neighbourhood:open",
            handler
        );

        onCleanup(() => {
            window.removeEventListener(
                "neighbourhood:open",
                handler
            );
        });
    });

    return (
        <svg
            width={props.width}
            height={props.height}
        >
            {/* center token */}
            <g>
                <circle
                    class={styles.ngCenter}
                    fill={
                        eeboStore.selected.color || "red"
                    }
                    cx={center().x}
                    cy={center().y}
                    r={16}
                />
                <text
                    class={styles.ngCenterText}
                    x="50%"
                    y="50%"
                    dy={4}
                >
                    {props.slice.token}
                </text>
                <text
                    class={styles.ngCenterTextSmall}
                    x="50%"
                    y="50%"
                    dy={30}
                >
                    {props.slice.slice_start}
                    -
                    {props.slice.slice_end}
                </text>
            </g>

            {/* neighbors */}
            <For each={layout()}>
                {(n) => (
                    <g>
                        <circle
                            class={styles.ngPoint}
                            fill={
                                eeboStore.selected.color!
                            }
                            cx={n.x}
                            cy={n.y}
                            r={nodeSize(n)}
                        />
                        <text
                            class={styles.ngPointText}
                            x={n.x}
                            y={n.y}
                            dy={4}
                        >
                            {n.token}
                        </text>
                    </g>
                )}
            </For>
        </svg>
    );
};

export default NeighborGraph;
