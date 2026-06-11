import { createSignal } from "solid-js";
import { fetchWindow } from "../services/tokenWindowApi";

// Cache persists for the lifetime of the component instance
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

    await Promise.all(
      missing.map(async (p) => {
        try {
          const text = await fetchWindow({ doc_id: p.doc_id, token_idx: p.token_idx });
          windowCache.set(p.event_id, text);
        } catch (e) {
          console.warn("[useClusterWindowCache] failed", p.event_id, e);
        }
      })
    );

    // Replace the signal value so consumers react
    setCache(new Map(windowCache));
    setLoading(prev => {
      const next = new Set(prev);
      missing.forEach(p => next.delete(p.event_id));
      return next;
    });
  }

  return { cache, loading, fetchForPoints };
}