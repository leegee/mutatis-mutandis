// src/lib/contextGraphUtils.ts
import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "../corpus_config";
import type {
    ConceptData, ContextNode, HubHubEdge, HubNbEdge, AnyEdge,
    ConceptEvent, TokenBin, ContextGraphData
} from "../types/context-graph.types";

export function scanYearRange(cd: ConceptData): [number, number] {
    let min = CORPUS_END_YEAR;
    let max = CORPUS_START_YEAR;
    for (const e of cd.events) {
        if (e.pub_year === undefined) continue;
        if (e.pub_year < min) min = e.pub_year;
        if (e.pub_year > max) max = e.pub_year;
    }
    return min <= max ? [min, max] : [CORPUS_START_YEAR, CORPUS_END_YEAR];
}

export function filterByYearRange(
    events: ConceptEvent[], from: number, to: number
): ConceptEvent[] {
    return events.filter(
        (e) => e.pub_year !== undefined && e.pub_year >= from && e.pub_year <= to
    );
}


/**
 * Aggregates contextual observations by lexical token.
 *
 * Events without a token field are skipped (no "__unknown__" fallback) so
 * that the hub set is clean.  Score sums are tracked separately from
 * frequency counts so meanScore is exact.
 */
export function aggregateByToken(events: ConceptEvent[]): Map<string, TokenBin> {
    const bins = new Map<string, TokenBin>();

    for (const event of events) {
        const binKey = event.token;
        // Skip events with no token — don't pollute hub set with unknowns.
        if (!binKey) continue;

        let bin = bins.get(binKey);
        if (!bin) {
            bin = {
                token: binKey,
                eventCount: 0,
                neighbourFreq: new Map(),
                neighbourScoreSum: new Map(),
                topNeighbours: [],
                docs: new Map(),
                years: new Set(),
            };
            bins.set(binKey, bin);
        }

        bin.eventCount += 1;
        if (event.doc_id) bin.docs.set(event.doc_id, event.pub_year);
        if (event.pub_year !== undefined) bin.years.add(event.pub_year);

        for (const nb of event.neighbours) {
            bin.neighbourFreq.set(nb.token, (bin.neighbourFreq.get(nb.token) ?? 0) + 1);
            bin.neighbourScoreSum.set(nb.token, (bin.neighbourScoreSum.get(nb.token) ?? 0) + nb.score);
        }
    }

    // Normalise frequencies and compute top neighbours.
    for (const bin of bins.values()) {
        const total = [...bin.neighbourFreq.values()].reduce((a, b) => a + b, 0);

        const normFreq = new Map<string, number>();
        for (const [tok, count] of bin.neighbourFreq)
            normFreq.set(tok, total > 0 ? count / total : 0);
        bin.neighbourFreq = normFreq;

        bin.topNeighbours = [...normFreq.entries()]
            .sort((a, b) => b[1] - a[1])
            .slice(0, 30)   // store top-30 so the drill-down has headroom
            .map(([tok, freq]) => {
                const rawCount = Math.round(freq * total); // recover raw count for division
                return {
                    token: tok,
                    freq,
                    meanScore: rawCount > 0
                        ? (bin.neighbourScoreSum.get(tok) ?? 0) / rawCount
                        : 0,
                };
            });
    }

    return bins;
}

/**
 * Sparse cosine similarity over neighbour-frequency vectors.
 * Iterates the smaller map for efficiency.
 */
export function cosineSimilarity(a: Map<string, number>, b: Map<string, number>): number {
    let normA = 0; for (const v of a.values()) normA += v * v;
    let normB = 0; for (const v of b.values()) normB += v * v;
    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);
    if (normA === 0 || normB === 0) return 0;
    const [smaller, larger] = a.size <= b.size ? [a, b] : [b, a];
    let dot = 0;
    for (const [tok, v] of smaller) {
        const u = larger.get(tok);
        if (u !== undefined) dot += v * u;
    }
    return Math.min(1, Math.max(0, dot / (normA * normB)));
}


/**
 * Build the aggregated two-kind graph (hub + neighbour nodes).
 */
export function buildContextualGraph(
    bins: Map<string, TokenBin>,
    topN: number,
    minSimilarity: number,
    maxHubs: number,
    EMPTY_GRAPH: any,
): ContextGraphData {
    const hubKeys = [...bins.keys()];
    if (hubKeys.length === 0) return EMPTY_GRAPH;

    // 1. Hub-hub edges: pairwise cosine -------------------------------------
    const rawHubHub: Array<[string, string, number]> = [];
    for (let i = 0; i < hubKeys.length; i++) {
        for (let j = i + 1; j < hubKeys.length; j++) {
            const sim = cosineSimilarity(
                bins.get(hubKeys[i])!.neighbourFreq,
                bins.get(hubKeys[j])!.neighbourFreq,
            );
            if (sim >= minSimilarity) rawHubHub.push([hubKeys[i], hubKeys[j], sim]);
        }
    }
    // TODO layer 2: inject temporal continuity edges here
    // TODO layer 3: inject shared-unusual-neighbour edges here

    // Hub degree from hub-hub edges only.
    const hubHubDegree = new Map<string, number>();
    for (const [a, b] of rawHubHub) {
        hubHubDegree.set(a, (hubHubDegree.get(a) ?? 0) + 1);
        hubHubDegree.set(b, (hubHubDegree.get(b) ?? 0) + 1);
    }

    // Retain all hubs but prefer the most contextually connected ones when
    // truncating: sort by hub-hub degree desc, then eventCount desc.
    const sortedHubs = hubKeys
        .sort((a, b) => {
            const dd = (hubHubDegree.get(b) ?? 0) - (hubHubDegree.get(a) ?? 0);
            if (dd !== 0) return dd;
            return bins.get(b)!.eventCount - bins.get(a)!.eventCount;
        })
        .slice(0, maxHubs);

    const hubSet = new Set(sortedHubs);

    // 2. Hub nodes---
    const nodeMap = new Map<string, ContextNode>();

    for (const key of sortedHubs) {
        nodeMap.set(key, {
            id: key,
            kind: "hub",
            eventCount: bins.get(key)!.eventCount,
            hubDegree: hubHubDegree.get(key) ?? 0,
            degree: hubHubDegree.get(key) ?? 0,
        });
    }

    // 3. Neighbour nodes + hub-neighbour edges -------------------------------
    const spokeTriples: Array<[string, string, number]> = [];

    for (const hubKey of sortedHubs) {
        const bin = bins.get(hubKey)!;
        for (const nb of bin.topNeighbours.slice(0, topN)) {
            if (!nodeMap.has(nb.token)) {
                nodeMap.set(nb.token, {
                    id: nb.token, kind: "neighbour",
                    eventCount: 0, hubDegree: 0, degree: 0,
                });
            }
            spokeTriples.push([hubKey, nb.token, nb.freq]);
        }
    }

    for (const [hubKey, nbToken] of spokeTriples) {
        nodeMap.get(hubKey)!.degree += 1;
        nodeMap.get(nbToken)!.degree += 1;
    }

    const nodes = [...nodeMap.values()];

    // 4. Materialise edges ---------------------------------------------------
    const hubHubEdges: HubHubEdge[] = rawHubHub
        .filter(([a, b]) => hubSet.has(a) && hubSet.has(b))
        .map(([a, b, weight]) => ({
            kind: "hub-hub" as const,
            source: nodeMap.get(a)!,
            target: nodeMap.get(b)!,
            weight,
        }));

    const hubNbEdges: HubNbEdge[] = spokeTriples.map(([hubKey, nbToken, weight]) => ({
        kind: "hub-neighbour" as const,
        source: nodeMap.get(hubKey)!,
        target: nodeMap.get(nbToken)!,
        weight,
    }));

    const allEdges: AnyEdge[] = [...hubNbEdges, ...hubHubEdges];

    const maxEventCount = Math.max(1, ...nodes.filter(n => n.kind === "hub").map(n => n.eventCount));
    const maxHubDegree = Math.max(1, ...nodes.filter(n => n.kind === "hub").map(n => n.hubDegree));
    const maxHubHubWeight = Math.max(1, ...hubHubEdges.map(e => e.weight));

    console.log("[ctx-graph] hubs:", sortedHubs.length,
        "| nb nodes:", nodes.filter(n => n.kind === "neighbour").length,
        "| hub-hub edges:", hubHubEdges.length,
        "| spoke edges:", hubNbEdges.length);

    return { nodes, hubHubEdges, hubNbEdges, allEdges, maxHubHubWeight, maxEventCount, maxHubDegree };
}


/**
 * Build a raw event graph: one node per ConceptEvent, linked to their top-N
 * neighbour tokens.  Neighbour nodes are shared across events.
 * No hub-hub edges.
 */
export function buildPureEventGraph(
    events: ConceptEvent[],
    topN: number,
    EMPTY_GRAPH: any
): ContextGraphData {
    if (events.length === 0) return EMPTY_GRAPH;

    const nodeMap = new Map<string, ContextNode>();
    const hubNbEdges: HubNbEdge[] = [];

    for (let idx = 0; idx < events.length; idx++) {
        const event = events[idx];
        const nodeId = event.event_id !== undefined
            ? `event_${ event.event_id }`
            : `event_idx:${ idx }`;

        const eventNode: ContextNode = {
            id: nodeId,
            kind: "event",
            eventCount: 1,
            hubDegree: 0,
            degree: 0,
            token: event.token,
            doc_id: event.doc_id,
            pub_year: event.pub_year,
        };
        nodeMap.set(nodeId, eventNode);

        const top = [...event.neighbours]
            .sort((a, b) => b.score - a.score)
            .slice(0, topN);

        for (const nb of top) {
            if (!nodeMap.has(nb.token)) {
                nodeMap.set(nb.token, {
                    id: nb.token, kind: "neighbour",
                    eventCount: 0, hubDegree: 0, degree: 0,
                });
            }
            const nbNode = nodeMap.get(nb.token)!;
            hubNbEdges.push({
                kind: "hub-neighbour" as const,
                source: eventNode,
                target: nbNode,
                weight: nb.score,
            });
            eventNode.degree += 1;
            nbNode.degree += 1;
        }
    }

    const nodes = [...nodeMap.values()];

    console.log("[ctx-graph/events] event nodes:", events.length,
        "| nb nodes:", nodes.filter(n => n.kind === "neighbour").length,
        "| spoke edges:", hubNbEdges.length);

    return {
        nodes,
        hubHubEdges: [],
        hubNbEdges,
        allEdges: hubNbEdges,
        maxHubHubWeight: 1,
        maxEventCount: 1,
        maxHubDegree: 1,
    };
}


