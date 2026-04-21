import type { Dataset, Slice, SliceView, TokenData } from "../types";
import { buildSliceHistory, detectTransitions } from "../signals/sliceSignals";
import SLICE_RANGES from "../services/SLICES.json";

export function buildSliceView(
    token: string,
    slice: Slice,
    dataset: Dataset
): SliceView {

    const tokenData = dataset[token];

    const neighbors = slice.top_neighbors ?? [];

    const rank = new Map(
        neighbors.map((n, i) => [n.token, { rank: i, mass: n.mass }])
    );

    // keep aligned raw history (with nulls)
    const rawHistory = SLICE_RANGES.map(([start, end]) => {
        const key = `${start}-${end}`;
        return tokenData?.[key as keyof TokenData] ?? null;
    });

    // build proper history points
    const historyPoints = buildSliceHistory(rawHistory);

    const detected = detectTransitions(historyPoints);

    const drift = slice.drift ?? 0;

    const maxDrift = Math.max(
        ...rawHistory.map(s => s?.drift ?? 0)
    );

    return {
        token,
        slice_start: slice.slice_start,
        slice_end: slice.slice_end ?? slice.slice_start,

        neighbors,
        drift,
        normalizedDrift: maxDrift > 0 ? drift / maxDrift : 0,

        rank,

        history: historyPoints,
        transitions: detected
    };
}
