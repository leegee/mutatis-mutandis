import { Show } from "solid-js";
import GraphWorkspace from "~/components/GraphWorkspace";
import { liveEntities, liveRelations } from "~/db/live";
import { useLiveQuery } from "~/db/useLiveQuery";
import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

export default function Home() {
	const entities = useLiveQuery(liveEntities(), [] as Entity[]);
	const relations = useLiveQuery(liveRelations(), [] as Relation[]);

	return (
		<section class="padding">
			<Show when={!entities.loading() && !relations.loading()} fallback={<progress />}>
				<GraphWorkspace entities={entities.value()} relations={relations.value()} />
			</Show>
		</section>
	);
}
