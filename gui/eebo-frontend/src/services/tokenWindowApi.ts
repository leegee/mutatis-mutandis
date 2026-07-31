import { createResource } from "solid-js";

export function createTokenWindowResource(
  event: () => { corpus: string, doc_id: string; token_idx: number } | null
) {
  return createResource(event, async (e) => {
    if (!e) return null;
    return fetchWindow(e);
  });
}

export async function fetchWindow(e: { corpus: string, doc_id: string; token_idx: number }) {
  console.log("[tokenWindowApi]", e);
  const res = await fetch(`/api/window/${ e.corpus }/${ e.doc_id }/${ e.token_idx }`);
  if (!res.ok) throw new Error("failed to fetch window");
  return res.text();
}
