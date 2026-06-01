# SQLite migration notes

## Files changed

| File | Status |
|------|--------|
| `src/services/db.ts` | **New** — sqlite-wasm singleton + query helpers |
| `src/state/tier2data.store.ts` | **Replaced** — loads `.db`, exposes `dbReady` / `dbError` signals |
| `src/state/selectors.ts` | **Replaced** — async SQL queries; sync helpers kept |
| `src/components/NeighbourhoodBrowser.tsx` | **Ported** — `createMemo` → `createResource` for data fetching |
| `src/types/context-graph.types.ts` | **Unchanged** |
| `src/lib/contextGraphUtils.ts` | **Unchanged** |
| `src/services/tokenWindowApi.ts` | **Unchanged** |
| `src/state/controls.store.ts` | **Unchanged** |

---

## Setup

### 1. Install sqlite-wasm

```bash
npm install @sqlite.org/sqlite-wasm
```

### 2. Vite config

sqlite-wasm ships a Worker and a `.wasm` file that must be served with the
correct headers and excluded from dependency optimisation:

```ts
// vite.config.ts
import { defineConfig } from "vite";

export default defineConfig({
  optimizeDeps: {
    exclude: ["@sqlite.org/sqlite-wasm"],
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
```

The COOP + COEP headers are required for OPFS (persistent storage).
Without them the database falls back to in-memory and is re-fetched on
every page load.

### 3. Add DB URL to corpus_config.ts

```ts
export const CORPUS_TIER2_DB_URL = "/data/tier2_concept_neighbours.db";
```

Serve the `.db` file as a static asset (e.g. drop it in `public/data/`).

### 4. App startup

Call `loadTier2Data()` once before rendering, e.g. in `App.tsx`:

```tsx
import { loadTier2Data } from "./state/tier2data.store";

loadTier2Data(); // fire and forget — dbReady() goes true when done

const App = () => (
  <Show when={dbReady()} fallback={<p>Loading database…</p>}>
    <NeighbourhoodBrowser />
  </Show>
);
```

---

## Architecture

```
fetch .db → OPFS (or memory fallback)
    ↓
db.ts  queryEvents(concept, from, to)
           SELECT events WHERE concept=? AND pub_year BETWEEN ? AND ?
           + SELECT neighbours WHERE event_id IN (...)   ← single join query
    ↓
selectors.ts  getYearFiltered() → ConceptEvent[]
    ↓
NeighbourhoodBrowser  createResource(resourceKey, getYearFiltered)
    ↓
buildNeighbourIndex()  ← unchanged pure function
    ↓
Left / Centre / Right panels  ← unchanged rendering
```

---

## Key design decisions

**Why `createResource` instead of `createMemo`?**
`getYearFiltered` is now async (SQL query). `createResource` is SolidJS's
primitive for async reactive data; it exposes `.loading` and error state
cleanly and integrates with `<Suspense>`.

**Why not stream rows?**
sqlite-wasm is synchronous inside the Wasm boundary. The async boundary is
only at the `fetch` call when loading the file. Once the DB is open, queries
are synchronous and fast — a year-filtered slice of 500k events returns in
milliseconds.

**Why inline the neighbour IDs in SQL rather than a subquery?**
sqlite-wasm's JS binding does not support array bind parameters. The inline
`IN (1,2,3,...)` pattern is safe here because the IDs are integers from our
own DB (no injection risk) and avoids a correlated subquery that SQLite's
planner may not flatten optimally.

**Why keep `filterEvents` in selectors.ts?**
`contextGraphUtils.ts` and other utilities that receive a `ConceptEvent[]`
already in memory still need it. It's a pure function over arrays so there
is no reason to push it into SQL.

---

## Other components that read tier2Data

Any component that previously read `tier2Data[controls.concept]` directly
from the store needs updating. The pattern is:

```tsx
// Before
const events = createMemo(() => tier2Data[controls.concept]?.events ?? []);

// After
const [eventsResource] = createResource(
  () => [controls.concept, controls.fromYear, controls.toYear] as const,
  ([c, f, t]) => getYearFiltered(c, f, t),
);
const events = () => eventsResource() ?? [];
```

`queryAggregate(concept)` in `db.ts` is available for components that
previously read `tier2Data[concept].aggregate`.