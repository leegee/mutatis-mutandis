// lib/tooltipPosition.ts

import type { JSX } from "solid-js";

export function computeTooltipStyle(
    x: number,
    y: number,
): JSX.CSSProperties {
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    const topThird = viewportHeight / 3;
    const rightThird = (viewportWidth * 2) / 3;

    const rv: JSX.CSSProperties = {
        position: "fixed",
        left: x > rightThird ? undefined : `${ x + 100 }px`,
        right: x > rightThird ? `${ viewportWidth - x - 100 }px` : undefined,
        top: y < topThird ? `${ y + 90 }px` : undefined,
        bottom: y < topThird ? undefined : `${ viewportHeight - y - 40 }px`,
    };

    // console.log("[tooltip-position]", x, y, rv);
    return rv;
}
