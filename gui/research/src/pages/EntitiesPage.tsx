import { For, Show } from "solid-js";
import EntityForm from "~/components/EntityForm";
import { liveEntities } from "~/db/live";
import { useLiveQuery } from "~/db/useLiveQuery";
import type { Entity } from "~/domain/entity";

export default function EntitiesPage() {
  const entities = useLiveQuery(liveEntities(), [] as Entity[]);

  return (
    <article class="small page active">
      <section class="padding">
        <h2>Add entity</h2>
        <EntityForm />
      </section>

      <section class="padding">
        <h2>Entities</h2>

        <Show when={!entities.loading()} fallback={<progress />}>
          <Show
            when={entities.value().length > 0}
            fallback={<p>No entities yet.</p>}
          >
            <ul class="list no-space border">
              <For each={entities.value()}>
                {(entity) => (
                  <li>
                    <strong>{entity.label}</strong>
                    {" — "}
                    {entity.type}
                  </li>
                )}
              </For>
            </ul>
          </Show>
        </Show>
      </section>
    </article >
  );
}
