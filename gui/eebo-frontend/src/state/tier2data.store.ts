// src/state/tier2data.store.ts

import { createStore } from "solid-js/store";

import type { Tier2Data } from "../types/context-graph.types";
import { CORPUS_TIER2_URL } from "../corpus_config";

export const [tier2Data, setTier2Data] = createStore<Tier2Data>({});

export async function loadTier2Data() {
  const res = await fetch(CORPUS_TIER2_URL);

  if (!res.ok) {
    throw new Error(`Failed to load semantic events: ${ res.status } ${ res.statusText }`);
  }

  const json = await res.json();
  setTier2Data(json as Tier2Data);
}
