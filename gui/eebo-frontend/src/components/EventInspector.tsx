import type { SemanticEvent } from "../types/events";

type Props = {
  event: SemanticEvent | null;
};

export default function EventInspector(props: Props) {
  const groupedNeighbours = () => {
    const ev = props.event;
    if (!ev) return [];

    const map = new Map<string, number>();

    for (const n of ev.neighbours) {
      // normalize similarity BEFORE grouping
      const sim = Math.round(n.similarity * 1000) / 1000;

      const key = `${ n.token }::${ sim }`;

      map.set(key, (map.get(key) ?? 0) + 1);
    }

    return Array.from(map.entries())
      .map(([key, count]) => {
        const [token, sim] = key.split("::");

        return {
          token,
          similarity: Number(sim),
          count
        };
      })
      .sort((a, b) => {
        // importance = similarity desc
        if (b.similarity !== a.similarity) {
          return b.similarity - a.similarity;
        }

        return b.count - a.count;
      });
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
          <p>Document: {event()!.doc_id}</p>

          <h5>Neighbours</h5>

          <ul class="list no-space border">
            {groupedNeighbours().map(n => (
              <li>
                {n.count} × {n.token}{" "}
                ({n.similarity.toFixed(3)})
              </li>
            ))}
          </ul>
        </>
      )}
    </article>
  );
}