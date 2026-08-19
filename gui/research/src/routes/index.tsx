import { Show } from "solid-js";
import GraphWorkspace from "~/components/GraphWorkspace";
import { liveEntities, liveEvidence, liveRelations } from "~/db/live";
import { useLiveQuery } from "~/db/useLiveQuery";
import type { Entity } from "~/domain/entity";
import type { Evidence } from "~/domain/evidence";
import type { Relation } from "~/domain/relation";

export default function Home() {
  const entities = useLiveQuery(liveEntities(), [] as Entity[]);
  const relations = useLiveQuery(liveRelations(), [] as Relation[]);
  const evidence = useLiveQuery(liveEvidence(), [] as Evidence[]);

  return (
    <section class="padding">
      <Show
        when={!entities.loading() && !relations.loading()}
        fallback={<progress />}
      >
        <GraphWorkspace
          entities={entities.value()}
          relations={relations.value()}
          evidence={evidence.value()}
        />
      </Show>
    </section>
  );
}
