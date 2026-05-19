import type { SemanticEvent } from "../types/events";

type Props = {
  event: SemanticEvent | null;
};

type Bucket = {
  range: string;
  count: number;
  maxSim: number;
};

type Grouped = {
  token: string;
  maxSimilarity: number;
  buckets: Bucket[];
};

export default function EventInspector(props: Props) {
  const groupedNeighbours = (): Grouped[] => {
    const ev = props.event;
    if (!ev) return [];

    const map = new Map<
      string,
      {
        maxSim: number;
        buckets: Map<string, { count: number; maxSim: number }>;
      }
    >();

    for (const n of ev.neighbours) {
      const token = n.token;

      const sim = n.similarity;
      const bucketStart = Math.floor(sim * 100) / 100;
      const bucketKey = `${ bucketStart.toFixed(2) }–${ (bucketStart + 0.01).toFixed(2) }`;

      if (!map.has(token)) {
        map.set(token, {
          maxSim: sim,
          buckets: new Map()
        });
      }

      const entry = map.get(token)!;

      // update token max similarity
      if (sim > entry.maxSim) {
        entry.maxSim = sim;
      }

      // update bucket
      const b = entry.buckets.get(bucketKey);

      if (!b) {
        entry.buckets.set(bucketKey, {
          count: 1,
          maxSim: sim
        });
      } else {
        b.count += 1;
        if (sim > b.maxSim) {
          b.maxSim = sim;
        }
      }
    }

    return Array.from(map.entries())
      .map(([token, v]) => ({
        token,
        maxSimilarity: v.maxSim,

        // 🔽 SORT BUCKETS by strongest similarity first
        buckets: Array.from(v.buckets.entries())
          .map(([range, b]) => ({
            range,
            count: b.count,
            maxSim: b.maxSim
          }))
          .sort((a, b) => b.maxSim - a.maxSim)
      }))

      // 🔽 SORT TOKENS by strongest similarity first
      .sort((a, b) => b.maxSimilarity - a.maxSimilarity);
  };

  const event = () => props.event;

  return (
    <article class="inspector">
      {event() && (
        <>
          <h3>
            <code>{event()!.token}</code>
          </h3>

          <p>Concept set: {event()!.concept}</p>
          <p>Slice: {event()!.slice}</p>
          <p>Vector ID: {event()!.vector_id}</p>
          <p>Document: <a target="_blank" href={`/xml/${ event()!.filepath }`}>{event()!.doc_id}</a></p>

          <h5>Neighbours</h5>

          <ul class="list no-space border">
            {groupedNeighbours().map(group => (
              <li>
                <strong>
                  {group.token} ({group.maxSimilarity.toFixed(3)})
                </strong>

                <ul class="list no-space no-margin">
                  {group.buckets.map(b => (
                    <li>
                      {b.count} × {b.range}
                    </li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        </>
      )}
    </article>
  );
}
