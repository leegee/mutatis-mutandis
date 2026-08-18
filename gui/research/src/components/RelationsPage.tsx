import {
  createEffect,
  createSignal,
  For,
  Show,
} from "solid-js";

import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

import {
  listEntities,
  listRelations,
} from "~/db/repository";

import RelationForm from "~/components/RelationForm";

export default function RelationsPage() {
  const [entities, setEntities] =
    createSignal<Entity[]>([]);

  const [relations, setRelations] =
    createSignal<Relation[]>([]);

  const [loading, setLoading] =
    createSignal(true);

  async function refresh() {
    setLoading(true);

    try {
      const [newEntities, newRelations] =
        await Promise.all([
          listEntities(),
          listRelations(),
        ]);

      setEntities(newEntities);
      setRelations(newRelations);
    } finally {
      setLoading(false);
    }
  }

  function entityLabel(id: string) {
    return (
      entities().find(
        (entity) => entity.id === id,
      )?.label ?? id
    );
  }

  createEffect(() => {
    if (typeof window !== "undefined") {
      refresh();
    }
  });

  return (
    <main class="responsive">
      <section class="large-padding">
        <h2>Add relationship</h2>

        <RelationForm
          onCreated={refresh}
        />
      </section>

      <section class="large-padding">
        <h2>Relationships</h2>

        <Show
          when={!loading()}
          fallback={<p>Loading...</p>}
        >
          <Show
            when={relations().length > 0}
            fallback={
              <p>No relationships yet.</p>
            }
          >
            <ul>
              <For each={relations()}>
                {(relation) => (
                  <li>
                    {entityLabel(
                      relation.sourceId,
                    )}

                    {" — "}

                    <strong>
                      {relation.type}
                    </strong>

                    {" → "}

                    {entityLabel(
                      relation.targetId,
                    )}
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
