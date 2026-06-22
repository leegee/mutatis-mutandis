import { createSignal, createEffect, Show, For } from "solid-js";
import { execRows } from "../services/db";
import { controls } from "../state/controls.store";
import ControlsHeader from "./ControlsHeader";

interface AggregateRow {
  concept: string;
  nEvents: number;
  category: string;
  rank: number;
  item: string;
  count: number;
}

export default function ConceptAggregates() {
  const [tokenRows, setTokenRows] = createSignal<AggregateRow[]>([]);
  const [docRows, setDocRows] = createSignal<AggregateRow[]>([]);
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);

  const runQuery = async (concept: string) => {
    if (!concept) {
      setTokenRows([]);
      setDocRows([]);
      setError(null);
      return;
    }

    setLoading(true);
    setError(null);

    const sql = `
      SELECT
          c.concept,
          c.n_events,
          CASE a.kind
              WHEN 'token'  THEN 'Top Token'
              WHEN 'doc'    THEN 'Top Document'
          END as category,
          a.rank,
          COALESCE(a.value, a.window_doc_id || ':' || COALESCE(a.window_id, '')) as item,
          a.count
      FROM concepts c
      JOIN concept_aggregate a ON c.concept = a.concept
      WHERE c.concept = ?
        AND a.kind IN ('token', 'doc')
      ORDER BY a.kind, a.rank;
    `;

    try {
      const rawRows = await execRows(sql, [concept]);

      const typedRows: AggregateRow[] = rawRows.map((row) => ({
        concept: String(row[0]),
        nEvents: Number(row[1]),
        category: String(row[2]),
        rank: Number(row[3]),
        item: String(row[4]),
        count: Number(row[5]),
      }));

      setTokenRows(typedRows.filter(r => r.category === 'Top Token'));
      setDocRows(typedRows.filter(r => r.category === 'Top Document'));
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Failed to execute query");
      setTokenRows([]);
      setDocRows([]);
    } finally {
      setLoading(false);
    }
  };

  createEffect(() => {
    const currentConcept = controls.concept;
    if (currentConcept) {
      runQuery(currentConcept);
    } else {
      setTokenRows([]);
      setDocRows([]);
      setError(null);
    }
  });

  return (
    <article class="concept-aggregates">
      <ControlsHeader />

      <h2>Concept Aggregates</h2>

      <Show when={loading()}>
        <p>
          Loading aggregates...
          <progress />
        </p>
      </Show>

      <Show when={error()}>
        <aside class="error-container"><h3>Error</h3>{error()}</aside>
      </Show>

      <div class="grid">

        {/* Tokens Table - Left Column */}
        <div class="s6">
          <Show when={tokenRows().length > 0 && !loading()}>
            <section>
              <h3>Top Tokens</h3>
              <div class="large-height scroll surface">
                <table class="stripes no-border scroll max">
                  <thead class="fixed">
                    <tr>
                      <th>Rank</th>
                      <th>Token</th>
                      <th>Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    <For each={tokenRows()}>
                      {(row) => (
                        <tr>
                          <td>{row.rank + 1}</td>
                          <td><strong>{row.item}</strong></td>
                          <td>{new Intl.NumberFormat().format(row.count)}</td>
                        </tr>
                      )}
                    </For>
                  </tbody>
                </table>
              </div>
            </section>
          </Show>
        </div>

        {/* Documents Table - Right Column with extra metrics */}
        <div class="s6">
          <Show when={docRows().length > 0 && !loading()}>
            <section>
              <h3>Top Documents</h3>
              <div class="large-height scroll surface">
                <table class="stripes no-border scroll max">
                  <thead class="fixed">
                    <tr>
                      <th>Rank</th>
                      <th>Document ID</th>
                      <th>Count</th>
                      <th>Avg per Event</th>
                      <th>% of Neighbours</th>
                    </tr>
                  </thead>
                  <tbody>
                    <For each={docRows()}>
                      {(row) => {
                        const avgPerEvent = row.nEvents > 0 ? (row.count / row.nEvents).toFixed(2) : "0.00";
                        const totalNeighbourSlots = row.nEvents * 25;
                        const percent = totalNeighbourSlots > 0
                          ? ((row.count / totalNeighbourSlots) * 100).toFixed(1)
                          : "0.0";

                        return (
                          <tr>
                            <td>{row.rank + 1}</td>
                            <td><strong>{row.item}</strong></td>
                            <td>{new Intl.NumberFormat().format(row.count)}</td>
                            <td>{avgPerEvent}</td>
                            <td>{percent}%</td>
                          </tr>
                        );
                      }}
                    </For>
                  </tbody>
                </table>
              </div>
            </section>
          </Show>
        </div>

      </div>

      <Show when={!controls.concept}>
        <p>Select a concept to view its neighbourhood aggregates.</p>
      </Show>

      <Show when={controls.concept && tokenRows().length === 0 && docRows().length === 0 && !loading()}>
        <p>No aggregates found for this concept.</p>
      </Show>
    </article>
  );
}