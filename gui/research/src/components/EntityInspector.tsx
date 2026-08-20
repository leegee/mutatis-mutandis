import { createEffect, createSignal, For, onCleanup, onMount, Show } from "solid-js";
import {
	addEntityAlias,
	addEntityTag,
	deleteEntity,
	listAliases,
	listTags,
	removeEntityAlias,
	removeEntityTag,
} from "~/db/respository";
import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";
import AutoComplete from "./AutoComplete";
import EntityForm from "./EntityForm";
import { useConfirm } from "./Modal";

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
	const [tagInput, setTagInput] = createSignal("");
	const [tags, setTags] = createSignal<string[]>([]);
	const [aliases, setAliases] = createSignal<string[]>([]);
	const [aliasInput, setAliasInput] = createSignal("");
	const confirm = useConfirm();

	createEffect(() => {
		listTags().then(setTags);
		listAliases().then(setAliases);
	});

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

	async function handleAddTag(tag: string) {
		const value = tag.trim();
		if (!value) return;

		const entity = currentEntity();
		const updated = await addEntityTag(entity, value);

		setCurrentEntity(updated);
		setTagInput("");

		// Refresh the global tag list in case this was a new tag.
		setTags(await listTags());

		await props.onChanged?.(updated);
	}

	async function handleRemoveTag(tag: string) {
		const entity = currentEntity();
		const updated = await removeEntityTag(entity, tag);
		setCurrentEntity(updated);
		await props.onChanged?.(updated);
	}

	async function handleAddAlias(alias: string) {
		const value = alias.trim();
		if (!value) return;
		const entity = currentEntity();
		const updated = await addEntityAlias(entity, value);
		setCurrentEntity(updated);
		setAliasInput("");
		await props.onChanged?.(updated);
	}

	async function handleRemoveAlias(alias: string) {
		const entity = currentEntity();
		const updated = await removeEntityAlias(entity, alias);
		setCurrentEntity(updated);
		await props.onChanged?.(updated);
	}

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
								<div class="max">
									<h2> {entity().label} </h2>
									<span> {entity().type} </span>
								</div>

								<button
									class="circle transparent"
									type="button"
									title="Close"
									onClick={() => props.onClose?.(entity())}
								>
									<i>close</i>
								</button>
							</nav>
						</header>

						<Show when={entity().description}>
							<section class="surface">
								<p> {entity().description} </p>
							</section>
						</Show>

						<section class="surface">
							<AutoComplete<string>
								value={aliasInput()}
								items={aliases()}
								getLabel={(alias) => alias}
								onInput={setAliasInput}
								onEnter={() => handleAddAlias(aliasInput())}
								onSelect={handleAddAlias}
								placeholder="Aliases"
								isTitle
							/>

							<Show when={currentEntity().aliases.length > 0} fallback={<p>No aliases.</p>}>
								<div class="row wrap tiny-space">
									<For each={currentEntity().aliases}>
										{(alias) => (
											<span class="small chip left-padding">
												{alias}

												<button
													type="button"
													class="transparent small circle no-padding"
													title={`Remove ${ alias }`}
													aria-label={`Remove alias ${ alias }`}
													onClick={() => handleRemoveAlias(alias)}
												>
													<i class="small">close</i>
												</button>
											</span>
										)}
									</For>
								</div>
							</Show>
						</section>

						<section class="surface">
							<AutoComplete<string>
								value={tagInput()}
								items={tags()}
								getLabel={(tag) => tag}
								onInput={setTagInput}
								onSelect={handleAddTag}
								onEnter={() => handleAddTag(tagInput())}
								placeholder="Tags"
								isTitle
							/>

							<Show when={currentEntity().tags.length > 0} fallback={<p>No tags.</p>}>
								<div class="row wrap tiny-space">
									<For each={currentEntity().tags}>
										{(tag) => (
											<span class="small chip left-padding">
												{tag}

												<button
													type="button"
													class="transparent small circle no-padding"
													title={`Remove ${ tag }`}
													aria-label={`Remove tag ${ tag }`}
													onClick={() => handleRemoveTag(tag)}
												>
													<i class="small">close</i>
												</button>
											</span>
										)}
									</For>
								</div>
							</Show>
						</section>

						<section class="surface">
							<h3>Relationships</h3>

							<Show
								when={outgoing().length > 0 || incoming().length > 0}
								fallback={<p>Right-click a node to estabish a relationship </p>}
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
							<button type="button" onClick={() => setEditing(true)}>
								Edit
							</button>

							<button type="button" class="error" onClick={handleDelete}>
								Delete
							</button>
						</nav>
					</Show>
				</aside>
			)}
		</Show>
	);
}
