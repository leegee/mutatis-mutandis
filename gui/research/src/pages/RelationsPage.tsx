import { For, Show } from "solid-js";
import RelationForm from "~/components/RelationForm";
import { liveEntities, liveRelations } from "~/db/live";
import { useLiveQuery } from "~/db/useLiveQuery";
import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

export default function RelationsPage() {
  const entities = useLiveQuery(liveEntities(), [] as Entity[]);

  const relations = useLiveQuery(liveRelations(), [] as Relation[]);

  function entityLabel(id: string) {
    return entities.value().find((entity) => entity.id === id)?.label ?? id;
  }

  return (
    <article class="top-level-view">
      <header class="padding">
        <h2>Relationships</h2>
      </header>

      <section class="padding">
        <RelationForm />
      </section>

      <section class="padding">
        <h3>All Relationships</h3>

        <Show
          when={!entities.loading() && !relations.loading()}
          fallback={<progress />}
        >
          <Show
            when={relations.value().length > 0}
            fallback={<p>No relationships yet.</p>}
          >
            <table class="small-height stripes surface scroll">
              <thead class="fixed">
                <tr>
                  <th>Subject</th>
                  <th>Relates to</th>
                  <th>Object</th>
                </tr>
              </thead>
              <For each={relations.value()}>
                {(relation) => (
                  <tr>
                    <td>
                      {entityLabel(relation.sourceId)}
                    </td>
                    <td>

                      <strong> {relation.type} </strong>

                    </td>
                    <td>
                      {entityLabel(relation.targetId)}
                    </td>
                  </tr>
                )}
              </For>
            </table>
          </Show>
        </Show>
      </section>
    </article>
  );
}
