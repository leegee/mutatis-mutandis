import { createMemo, type Component } from "solid-js";
import type { SliceView } from "../types";
import styles from './NeighborGraph.module.css';

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

        // normalize similarity (tune if needed)
        const simScaled = (sim - 0.6) / 0.4;
        const clamped = Math.max(0, Math.min(1, simScaled));

        const spread = 1 - Math.pow(clamped, 2);

        const countWeight = Math.log1p(d.count ?? 1) / 4;

        const radius =
            minRadius +
            (maxRadius - minRadius) * spread * (1 + countWeight);

        const angle =
            (i / n) * 2 * Math.PI +
            ((d.count ?? 0) % 7) * 0.03;

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
    const neighbors = createMemo<Neighbor[]>(() => {
        return props.slice.neighbors ?? [];
    });

    const layout = createMemo<PositionedNeighbor[]>(() => {
        const cx = props.width / 2;
        const cy = props.height / 2;

        return radialLayout(cx, cy, neighbors());
    });

    const center = createMemo(() => ({
        x: props.width / 2,
        y: props.height / 2
    }));

    return (
        <svg width={props.width} height={props.height}>
            {/* center token */}
            <g>
                <circle
                    class={styles.ngCenter}
                    cx={center().x}
                    cy={center().y}
                    r={16}
                />
                <text
                    class={styles.ngCenterText}
                    x="50%" y="50%"
                    dy={4}
                >
                    {props.slice.token}
                </text>
            </g>

            {/* neighbors */}
            {layout().map((n) => (
                <g>
                    <circle
                        class={styles.ngPoint}
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
            ))}
        </svg>
    );
};

export default NeighborGraph;