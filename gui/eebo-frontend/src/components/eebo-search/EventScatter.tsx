// src/components/EventScatter.tsx

import { createEffect, createMemo } from "solid-js";
import type { SemanticEvent } from "../../types/events";
import {
    hoveredEventId,
    selectedEventId,
    setHoveredEventId,
    setSelectedEventId
} from "../../state/selection";
import "./EventScatter.css";

type Props = {
    events: SemanticEvent[];
};

const WIDTH = 900;
const HEIGHT = 650;
const PAD = 0.06;

export default function EventScatter(props: Props) {
    const bounds = createMemo(() => {

        const xs = props.events.map(e => e.x);
        const ys = props.events.map(e => e.y);

        const minXRaw = Math.min(...xs);
        const maxXRaw = Math.max(...xs);
        const minYRaw = Math.min(...ys);
        const maxYRaw = Math.max(...ys);


        const dx = maxXRaw - minXRaw;
        const dy = maxYRaw - minYRaw;

        return {
            minX: minXRaw - dx * PAD,
            maxX: maxXRaw + dx * PAD,
            minY: minYRaw - dy * PAD,
            maxY: maxYRaw + dy * PAD
        };
    });

    // stable scaling (dataset-relative, NOT view-relative)

    const scaleX = (x: number) => {
        const { minX, maxX } = bounds();
        return ((x - minX) / (maxX - minX + 1e-9)) * WIDTH;
    };

    const scaleY = (y: number) => {
        const { minY, maxY } = bounds();
        return ((y - minY) / (maxY - minY + 1e-9)) * HEIGHT;
    };

    createEffect(() => {
        console.log(props)
    })

    return (
        <svg id="scatterplot"
            viewBox={`0 0 ${ WIDTH } ${ HEIGHT }`}
            preserveAspectRatio="xMidYMid meet"
        >
            {props.events.map(e => {
                const selected = selectedEventId() === e.id;
                const hovered = hoveredEventId() === e.id;
                return (
                    <circle
                        cx={scaleX(e.x)}
                        cy={scaleY(e.y)}

                        r={
                            selected
                                ? 10
                                : hovered
                                    ? 7
                                    : 5
                        }

                        classList={{
                            selected,
                            hovered,
                            dimmed: !!selectedEventId() && !selected
                        }}

                        data-concept={e.concept}
                        data-slice={e.slice}

                        onMouseEnter={() =>
                            setHoveredEventId(e.id)
                        }

                        onMouseLeave={() =>
                            setHoveredEventId(null)
                        }

                        onClick={() =>
                            setSelectedEventId(e.id)
                        }
                    />
                );
            })}
        </svg>
    );
}