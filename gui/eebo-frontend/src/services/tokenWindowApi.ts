import { createResource } from "solid-js";
import type { ConceptEvent } from "../types";

export function createTokenWindowResource(
  event: () => ConceptEvent | null
) {
  return createResource(event, async (e) => {
    if (!e) return null;

    console.log("[tokenWindowApi]", e)

    const res = await fetch(
      `/api/window/${ e.doc_id }/${ e.token_idx }`
    );

    if (!res.ok) throw new Error("failed to fetch window");

    return res.text();
  });
}
