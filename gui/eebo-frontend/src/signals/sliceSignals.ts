import type { Slice, SliceHistoryPoint } from "../types";

export type SliceSignalConfig = {
    transitionMultiplier?: number;
};

export function buildSliceHistory(
    slices: Slice[]
): SliceHistoryPoint[] {
    // invariant: slices must be time-ordered upstream
    const raw = slices.map(s => ({
        t: s.slice_start,
        drift: s.drift ?? 0
    }));

    const withD1 = raw.map((d, i) => ({
        ...d,
        d1: i === 0 ? 0 : d.drift - raw[i - 1].drift
    }));

    return withD1.map((d, i) => ({
        ...d,
        d2: i < 2 ? 0 : d.d1 - withD1[i - 1].d1
    }));
}

export function detectTransitions(
    history: SliceHistoryPoint[],
    multiplier = 2
): number[] {
    const d2Values = history.map(d => d.d2);

    const mean =
        d2Values.reduce((a, b) => a + b, 0) / Math.max(1, d2Values.length);

    const variance =
        d2Values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) /
        Math.max(1, d2Values.length);

    const std = Math.sqrt(variance);

    const threshold = std * multiplier;

    const transitions = history
        .filter(d => Math.abs(d.d2) > threshold)
        .map(d => d.t);

    return transitions;
}
