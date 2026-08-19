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
    <article class="small page active">
      <section class="padding">
        <h2>Add relationship</h2>
        <RelationForm />
      </section>

      <section class="padding">
        <h2>Relationships</h2>

        <Show
          when={!entities.loading() && !relations.loading()}
          fallback={<progress />}
        >
          <Show
            when={relations.value().length > 0}
            fallback={<p>No relationships yet.</p>}
          >
            <ul class="list no-space border">
              <For each={relations.value()}>
                {(relation) => (
                  <li>
                    {entityLabel(relation.sourceId)}

                    {" — "}

                    <strong> {relation.type} </strong>

                    {" → "}

                    {entityLabel(relation.targetId)}
                  </li>
                )}
              </For>
            </ul>
          </Show>
        </Show>
      </section>
    </article>
  );
}
