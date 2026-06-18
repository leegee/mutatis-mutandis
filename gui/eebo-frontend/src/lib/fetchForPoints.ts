import { createSignal } from "solid-js";
import { fetchWindowBatch } from "../services/tokenWindowBatchApi";

const windowCache = new Map<string, string>();

export function useClusterWindowCache() {
  const [cache, setCache] = createSignal<Map<string, string>>(new Map(windowCache));
  const [loading, setLoading] = createSignal<Set<string>>(new Set());

  async function fetchForPoints(
    points: { event_id: string; doc_id: string; token_idx: number }[]
  ) {
    const missing = points.filter(p => !windowCache.has(p.event_id));
    if (!missing.length) return;

    setLoading(prev => new Set([...prev, ...missing.map(p => p.event_id)]));

    try {
      const res = await fetchWindowBatch(
        missing.map(p => ({
          docId: p.doc_id,
          tokenIdx: p.token_idx,
        }))
      );

      // map results back using local event_id list
      for (const r of res.results) {
        // find matching event(s)
        const match = missing.find(
          p => p.doc_id === r.docId && p.token_idx === r.tokenIdx
        );

        if (match) {
          windowCache.set(match.event_id, r.content);
        }
      }
    } catch (e) {
      console.warn("[useClusterWindowCache] batch failed", e);
    }

    setCache(new Map(windowCache));

    setLoading(prev => {
      const next = new Set(prev);
      missing.forEach(p => next.delete(p.event_id));
      return next;
    });
  }

  return { cache, loading, fetchForPoints };
}
