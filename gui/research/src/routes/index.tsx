import { createEffect, createSignal } from "solid-js";

import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

import {
  listEntities,
  listRelations,
} from "~/db/repository";

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

  createEffect(() => {
    if (typeof window !== "undefined") {
      refresh();
    }
  });

  return (
    <section class="large-padding">
      <h2>Research map</h2>

      {loading() ? (
        <p>Loading...</p>
      ) : (
        <GraphWorkspace
          entities={entities()}
          relations={relations()}
          onChanged={refresh}
        />
      )}
    </section>
  );
}
