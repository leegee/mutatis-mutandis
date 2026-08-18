import {
  createSignal,
  onMount,
  For,
  Show,
} from "solid-js";

import type { Entity } from "~/domain/entity";

import { listEntities } from "~/db/repository";

import EntityForm from "~/components/EntityForm";

export default function EntitiesPage() {
  const [entities, setEntities] =
    createSignal<Entity[]>([]);

  const [loading, setLoading] =
    createSignal(true);

  async function refresh() {
    setLoading(true);

    try {
      setEntities(await listEntities());
    } finally {
      setLoading(false);
    }
  }

  onMount(refresh);

  return (
    <main class="responsive">
      <section class="large-padding">
        <h2>Add entity</h2>

        <EntityForm
          onCreated={refresh}
        />
      </section>

      <section class="large-padding">
        <h2>Entities</h2>

        <Show
          when={!loading()}
          fallback={<p>Loading...</p>}
        >
          <Show
            when={entities().length > 0}
            fallback={
              <p>No entities yet.</p>
            }
          >
            <ul>
              <For each={entities()}>
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
    </main>
  );
}

