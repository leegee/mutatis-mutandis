import { For, Show } from "solid-js";

import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";
import { liveEntities, liveRelations, } from "~/db/live";
import { useLiveQuery } from "~/db/useLiveQuery";
import RelationForm from "~/components/RelationForm";

export default function RelationsPage() {
  const entities = useLiveQuery(
    liveEntities(),
    [] as Entity[],
  );

  const relations = useLiveQuery(
    liveRelations(),
    [] as Relation[],
  );

  function entityLabel(id: string) {
    return (
      entities.value().find(
        (entity) => entity.id === id,
      )?.label ?? id
    );
  }

  return (
    <main class="responsive">
      <section class="large-padding">
        <h2>Add relationship</h2>

        <RelationForm />
      </section>

      <section class="large-padding">
        <h2>Relationships</h2>

        <Show when={!entities.loading() && !relations.loading()}
          fallback={<p>Loading...</p>}
        >
          <Show when={relations.value().length > 0}
            fallback={
              <p>
                No relationships yet.
              </p>
            }
          >
            <ul>
              <For each={relations.value()} >
                {(relation) => (
                  <li>
                    {entityLabel(relation.sourceId,)}

                    {" — "}

                    <strong> {relation.type} </strong>

                    {" → "}

                    {entityLabel(relation.targetId,)}
                  </li>
                )}
              </For>
            </ul>
          </Show>
        </Show>
      </section>
    </main>
  );
}
