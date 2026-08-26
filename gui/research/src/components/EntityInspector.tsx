import { createEffect, createSignal, For, onCleanup, onMount, Show } from "solid-js";
import { deleteEntity } from "~/db/respository";
import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";
import AutoComplete from "./AutoComplete";
import EntityAliases from "./EntityAliases";
import EntityForm from "./EntityForm";
import EntityTags from "./EntityTags";
import { useConfirm } from "./Modal";

const no_data_fallback_class = "bottom-padding no-margin center-align";

interface EntityInspectorProps {
	entity: Entity | undefined;
	entities: Entity[];
	relations: Relation[];

	onChanged?: (entity: Entity) => void | Promise<void>;
	onClose?: (entity: Entity) => void;
}

export default function EntityInspector(props: EntityInspectorProps) {
	const [editing, setEditing] = createSignal(false);
	const [currentEntity, setCurrentEntity] = createSignal<Entity>(props.entity!);
	const confirm = useConfirm();

	createEffect(() => {
		if (props.entity) {
			setCurrentEntity(props.entity);
		}
	});

	onMount(() => {
		const handleKeyDown = (event: KeyboardEvent) => {
			if (event.key !== "Escape") return;
			if (editing()) return;
			const entity = props.entity;
			if (entity) props.onClose?.(entity);
		};

		window.addEventListener("keydown", handleKeyDown);
		onCleanup(() => window.removeEventListener("keydown", handleKeyDown));
	});

	function entityLabel(id: string): string {
		return props.entities.find((entity) => entity.id === id)?.label ?? id;
	}

	function outgoing(): Relation[] {
		const entity = props.entity;
		if (!entity) return [];

		return props.relations.filter((relation) => relation.sourceId === entity.id);
	}

	function incoming(): Relation[] {
		const entity = props.entity;
		if (!entity) return [];

		return props.relations.filter((relation) => relation.targetId === entity.id);
	}

	async function handleDelete() {
		const entity = props.entity;
		if (!entity) return;

		const ok = await confirm(`Delete "${ entity.label }"?`);
		if (!ok) return;

		await deleteEntity(entity.id);
		await props.onChanged?.(entity);
		props.onClose?.(entity);
	}

	return (
		<Show when={props.entity} fallback={""}>
			{(entity) => (
				<aside class="padding surface-container">
					<Show
						when={!editing()}
						fallback={
							<EntityForm
								entity={entity()}
								onUpdated={async (updated: Entity) => {
									setEditing(false);
									await props.onChanged?.(updated);
								}}
								onCancel={() => setEditing(false)}
							/>
						}
					>
						{/* NORMAL INSPECTOR VIEW */}
						<header class="fixed surface top-padding" style="top:0">
							<nav>
								<button
									class="circle transparent top-align"
									type="button"
									title="Close"
									onClick={() => props.onClose?.(entity())}
								>
									<i>arrow_back</i>
								</button>

								<div class="max">
									<h2> {entity().label} </h2>
									<span> {entity().type} </span>
								</div>
							</nav>
						</header>

						<Show when={entity().description}>
							<section class="surface">
								<p> {entity().description} </p>
							</section>
						</Show>

						<EntityAliases
							entity={currentEntity()}
							onChanged={async (updated) => {
								setCurrentEntity(updated);
								await props.onChanged?.(updated);
							}}
						/>

						<EntityTags
							entity={currentEntity()}
							onChanged={async (updated) => {
								setCurrentEntity(updated);
								await props.onChanged?.(updated);
							}}
						/>

						<section class="surface">
							<h3>Relationships</h3>

							<Show
								when={outgoing().length > 0 || incoming().length > 0}
								fallback={<p class={no_data_fallback_class}>Right-click a node to estabish a relationship </p>}
							>
								<Show when={outgoing().length > 0}>
									<h4>Outgoing</h4>
									<ul class="list no-space border">
										<For each={outgoing()}>
											{(relation) => (
												<li>
													{relation.type}
													{" → "}
													{entityLabel(relation.targetId)}
												</li>
											)}
										</For>
									</ul>
								</Show>

								<Show when={incoming().length > 0}>
									<h4>Incoming</h4>
									<ul class="list no-space border">
										<For each={incoming()}>
											{(relation) => (
												<li>
													{relation.type}
													{" ← "}
													{entityLabel(relation.sourceId)}
												</li>
											)}
										</For>
									</ul>
								</Show>
							</Show>
						</section>

						<nav class="footer">
							<button type="button" class="error" onClick={handleDelete}>
								Delete
							</button>

							<button type="button" onClick={() => setEditing(true)}>
								Edit
							</button>
						</nav>
					</Show>
				</aside>
			)}
		</Show>
	);
}
