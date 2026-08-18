import { createEffect, createSignal, For, Show } from "solid-js";

import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

import {
  listEntities,
  listRelations,
} from "~/db/repository";

import EntityForm from "~/components/EntityForm";
import RelationForm from "~/components/RelationForm";
import ProjectImport from "~/components/ProjectImport";
import GraphWorkspace from "~/components/GraphWorkspace";

export default function Home() {
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
    <>
      <nav>
        <h1 class="max">Research Map</h1>

        <ProjectImport
          onImported={refresh}
        />
      </nav>

      <section>
        <h2>Add entity</h2>

        <EntityForm
          onCreated={refresh}
        />
      </section>

      <section>
        <h2>Add relationship</h2>

        <RelationForm
          onCreated={refresh}
        />
      </section>

      <section>
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

      <section>
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

      <section class="large-padding">
        <h2>Research map</h2>

        <GraphWorkspace
          entities={entities()}
          relations={relations()}
          onChanged={refresh}
        />
      </section>
    </>
  );
}