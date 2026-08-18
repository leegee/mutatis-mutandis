import { createEffect, createSignal, For, Show } from "solid-js";

import type { Entity } from "~/domain/entity";
import { listEntities } from "~/db/repository";
import EntityForm from "~/components/EntityForm";
import RelationForm from "~/components/RelationForm";
import { listRelations } from "~/db/repository";
import type { Relation } from "~/domain/relation";
import ProjectImport from "~/components/ProjectImport";
import GraphWorkspace from "~/components/GraphWorkspace";

export default function Home() {
  const [entities, setEntities] = createSignal<Entity[]>([]);
  const [loading, setLoading] = createSignal(true);
  const [relations, setRelations] = createSignal<Relation[]>([]);
  const [relationsLoading, setRelationsLoading] = createSignal(true);

  async function refreshRelations() {
    setRelationsLoading(true);

    try {
      setRelations(await listRelations());
    } finally {
      setRelationsLoading(false);
    }
  }

  async function refreshEntities() {
    setLoading(true);

    try {
      setEntities(await listEntities());
    } finally {
      setLoading(false);
    }
  }

  function entityLabel(id: string) {
    return entities().find((entity) => entity.id === id)?.label ?? id;
  }

  createEffect(() => {
    if (typeof window !== "undefined") {
      refreshEntities();
      refreshRelations();
    }
  });

  return (
    <>
      <nav>
        <h1 class="max">Research Map</h1>
        <ProjectImport
          onImported={async () => {
            await refreshEntities();
            await refreshRelations();
          }}
        />
      </nav>

      <section>
        <h2>Add entity</h2>
        <EntityForm onCreated={refreshEntities} />
      </section>

      <section>
        <h2>Add relationship</h2>

        <RelationForm onCreated={refreshRelations} />
      </section>

      <section>
        <h2>Relationships</h2>

        <Show
          when={!relationsLoading()}
          fallback={<p>Loading...</p>}
        >
          <Show
            when={relations().length > 0}
            fallback={<p>No relationships yet.</p>}
          >
            <ul>
              <For each={relations()}>
                {(relation) => (
                  <li>
                    {entityLabel(relation.sourceId)}
                    {" — "}
                    <strong>{relation.type}</strong>
                    {" → "}
                    {entityLabel(relation.targetId)}
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
            fallback={<p>No entities yet.</p>}
          >
            <ul>
              <For each={entities()}>
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

      <section class="large-padding">
        <h2>Research map</h2>

        <GraphWorkspace
          entities={entities()}
          relations={relations()}
        />
      </section>

    </>
  );
}
