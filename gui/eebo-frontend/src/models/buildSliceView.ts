import type { Dataset, Slice, SliceView } from "../types";
import { buildSliceHistory, detectTransitions } from "../signals/sliceSignals";

export function buildSliceView(
    token: string,
    slice: Slice,
    dataset: Dataset
): SliceView {
    const tokenData = dataset[token];

    const neighbors = slice.top_neighbors ?? [];

    const rank = new Map(
        neighbors.map((n, i) => [n.token, i])
    );

    const history = buildSliceHistory(tokenData.slices);

    const transitions = detectTransitions(history);

    const drift = slice.drift ?? 0;

    return {
        token,
        slice_start: slice.slice_start,
        slice_end: slice.slice_end ?? slice.slice_start,

        neighbors,
        drift,
        normalizedDrift: Math.min(1, drift / 5),

        rank,
        history,
        transitions
    };
}
