import { createEffect, createSignal, For, onCleanup, onMount, Show } from "solid-js";
import { deleteEntity, updateEntity } from "~/db/respository";
import type { Entity, EntityType } from "~/domain/entity";
import { entityTypes } from "~/domain/entity";
import type { Relation } from "~/domain/relation";
import EntityAliases from "./EntityAliases";
import EntityAutocomplete from "./EntityAutocomplete";
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
	const confirm = useConfirm();

	const [currentEntity, setCurrentEntity] = createSignal<Entity>(props.entity!);

	const [label, setLabel] = createSignal("");
	const [type, setType] = createSignal<EntityType>("concept");
	const [description, setDescription] = createSignal("");

	const [saving, setSaving] = createSignal(false);

	createEffect(() => {
		const entity = props.entity;

		if (!entity) return;

		setCurrentEntity(entity);
		setLabel(entity.label);
		setType(entity.type);
		setDescription(entity.description ?? "");
	});

	onMount(() => {
		const handleKeyDown = (event: KeyboardEvent) => {
			if (event.key !== "Escape") return;

			const entity = props.entity;
			if (entity) props.onClose?.(entity);
		};

		window.addEventListener("keydown", handleKeyDown);

		onCleanup(() => {
			window.removeEventListener("keydown", handleKeyDown);
		});
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

	async function saveEntity() {
		const entity = props.entity;
		if (!entity || saving()) return;

		const value = label().trim();
		if (!value) return;

		setSaving(true);

		try {
			const updated = await updateEntity(entity, {
				label: value,
				type: type(),
				description: description().trim(),
			});

			setCurrentEntity(updated);
			setLabel(updated.label);
			setType(updated.type);
			setDescription(updated.description ?? "");

			await props.onChanged?.(updated);
		} finally {
			setSaving(false);
		}
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
				<aside class="surface-container padding top-margin">
					{/* Header */}
					<header class="fixed surface top-padding" style="top:0">
						<nav class="no-padding bottom-margin top-align">
							<button
								class="circle transparent top-margin small-margin"
								type="button"
								title="Close"
								onClick={() => props.onClose?.(entity())}
							>
								<i>close</i>
							</button>

							<div class="max">
								<h2>{entity().label}</h2>
								<span>{entity().type}</span>
							</div>
						</nav>
					</header>

					{/* Editable entity fields */}
					<section class="surface-container padding">
						<EntityAutocomplete
							value={label()}
							onInput={(value) => setLabel(value)}
							onSelect={(selected) => {
								setLabel(selected.label);
								setType(selected.type);
								setDescription(selected.description ?? "");
							}}
							disabled={saving()}
						/>

						<div class="field border">
							<select
								value={type()}
								disabled={saving()}
								onChange={(event) => setType(event.currentTarget.value as EntityType)}
							>
								<For each={entityTypes}>{(entityType) => <option value={entityType}>{entityType}</option>}</For>
							</select>

							<output>Entity Type</output>
						</div>

						<div class="field textarea border">
							<textarea
								value={description()}
								disabled={saving()}
								onInput={(event) => setDescription(event.currentTarget.value)}
								rows={4}
							/>

							<output>Description</output>
						</div>

						<nav class="right-align">
							<button class="small" type="button" disabled={saving() || !label().trim()} onClick={saveEntity}>
								{saving() ? "Saving…" : "Update"}
							</button>
						</nav>
					</section>

					{/* Aliases */}
					<EntityAliases
						entity={currentEntity()}
						onChanged={async (updated) => {
							setCurrentEntity(updated);
							await props.onChanged?.(updated);
						}}
					/>

					{/* Tags */}
					<EntityTags
						entity={currentEntity()}
						onChanged={async (updated) => {
							setCurrentEntity(updated);
							await props.onChanged?.(updated);
						}}
					/>

					{/* Relationships */}
					<section class="surface-container top-padding top-margin">
						<h3>Relationships</h3>

						<Show
							when={outgoing().length > 0 || incoming().length > 0}
							fallback={<p class={no_data_fallback_class}>Right-click a node to establish a relationship</p>}
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

					{/* Delete */}
					<nav class="footer">
						<button type="button" class="error" onClick={handleDelete}>
							Delete
						</button>
					</nav>
				</aside>
			)}
		</Show>
	);
}
