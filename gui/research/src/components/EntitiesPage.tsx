import { For, Show } from "solid-js";

import type { Entity } from "~/domain/entity";
import { liveEntities } from "~/db/live";
import { useLiveQuery } from "~/db/useLiveQuery";
import EntityForm from "~/components/EntityForm";

export default function EntitiesPage() {
  const entities = useLiveQuery(
    liveEntities(),
    [] as Entity[],
  );

  return (
    <article>
      <section class="large-padding">
        <h2>Add entity</h2>

        <EntityForm />
      </section>

      <section class="large-padding">
        <h2>Entities</h2>

        <Show when={!entities.loading()}
          fallback={<p>Loading...</p>}
        >
          <Show when={entities.value().length > 0}
            fallback={
              <p>No entities yet.</p>
            }
          >
            <ul>
              <For each={entities.value()}>
                {(entity) => (
                  <li>
                    <strong>
                      {entity.label}
                    </strong>
                    {" — "}
                    {entity.type}
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
