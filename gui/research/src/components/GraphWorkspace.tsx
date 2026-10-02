import { createSignal, onCleanup, onMount, Show } from "solid-js";

import EntityInspector from "~/components/EntityInspector";
import GraphView from "~/components/GraphView";
import { Modal } from "~/components/Modal";
import RelationForm from "~/components/RelationForm";
import RelationInspector from "~/components/RelationInspector";

import { deleteEntity, deleteRelation } from "~/db/respository";

import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

import EntityForm from "./EntityForm";
import { useModal } from "./Modal";

interface GraphWorkspaceProps {
	entities: Entity[];
	relations: Relation[];
}

export default function GraphWorkspace(props: GraphWorkspaceProps) {
	const [selectedEntity, setSelectedEntity] = createSignal<Entity>();
	const [editingEntity, setEditingEntity] = createSignal(false);
	const [selectedRelation, setSelectedRelation] = createSignal<Relation>();
	const [editingRelation, setEditingRelation] = createSignal(false);

	const [addingRelation, setAddingRelation] = createSignal<{
		source: Entity;
		target: Entity;
	}>();

	const modal = useModal();

	onMount(() => {
		const handleKeyDown = (event: KeyboardEvent) => {
			if (event.key !== "Escape") return;

			if (selectedEntity()) {
				setSelectedEntity(undefined);
				setEditingEntity(false);
			}

			if (selectedRelation()) {
				setSelectedRelation(undefined);
				setEditingRelation(false);
			}
		};

		window.addEventListener("keydown", handleKeyDown);

		onCleanup(() => {
			window.removeEventListener("keydown", handleKeyDown);
		});
	});

	function handleSelectEntity(entity: Entity) {
		setSelectedEntity(entity);
		setEditingEntity(false);
	}
	function handleEditEntity(entity: Entity) {
		setSelectedEntity(entity);
		setEditingEntity(true);
	}

	// function handleSelectRelation(relation: Relation) {
	// 	setSelectedRelation(relation);
	// 	setEditingRelation(false);
	// }

	function handleEditRelation(relation: Relation) {
		setSelectedRelation(relation);
		setEditingRelation(true);
	}

	async function handleAddEntity() {
		await modal(
			(close) => (
				<EntityForm
					onCreated={(entity: Entity) => {
						setSelectedEntity(entity);
						setSelectedRelation(undefined);
						close();
					}}
					onCancel={close}
				/>
			),
			"Add entity",
		);
	}

	async function handleDeleteEntity(entity: Entity) {
		await deleteEntity(entity.id);

		if (selectedEntity()?.id === entity.id) {
			setSelectedEntity(undefined);
		}
	}

	function handleAddRelation(sourceId: string, targetId: string) {
		const source = props.entities.find((entity) => entity.id === sourceId);
		const target = props.entities.find((entity) => entity.id === targetId);
		if (!source || !target) return;
		setAddingRelation({ source, target });
	}

	function handleCreatedRelation(relation: Relation) {
		setAddingRelation(undefined);
		setSelectedRelation(relation);
		setSelectedEntity(undefined);
	}

	function handleCancelAddRelation() {
		setAddingRelation(undefined);
	}

	async function handleDeleteRelation(relation: Relation) {
		await deleteRelation(relation.id);
		if (selectedRelation()?.id === relation.id) {
			setSelectedRelation(undefined);
		}
	}

	async function handleEntityChanged(entity: Entity) {
		setSelectedEntity(entity);
	}

	return (
		<>
			<div style={{ "min-width": "0" }}>
				<GraphView
					entities={props.entities}
					relations={props.relations}
					onSelectEntity={handleSelectEntity}
					onSelectRelation={(relation) => {
						setSelectedRelation(relation);
						setSelectedEntity(undefined);
					}}
					onAddEntity={handleAddEntity}
					onEditEntity={handleEditEntity}
					onDeleteEntity={handleDeleteEntity}
					onAddRelation={handleAddRelation}
					onEditRelation={handleEditRelation}
					onDeleteRelation={handleDeleteRelation}
				/>
			</div>

			<Show when={selectedEntity() || selectedRelation()}>
				<div class="transparent" style={{
					position: "fixed",
					right: "1em",
					"overflow-y": "auto",
					"min-width": "30rem",
					"max-width": "50vw",
				}}>
					<Show when={selectedEntity()}
						fallback={
							<RelationInspector
								relation={selectedRelation()}
								entities={props.entities}
								editing={editingRelation()}
								onClose={() => {
									setSelectedRelation(undefined);
									setEditingRelation(false);
								}}
							/>
						}
					>
						{(entity) => (
							<EntityInspector
								entity={entity()}
								entities={props.entities}
								relations={props.relations}
								onChanged={handleEntityChanged}
								editing={editingEntity()}
								onClose={() => {
									setSelectedEntity(undefined);
									setEditingEntity(false);
								}}
							/>
						)}
					</Show>
				</div>
			</Show>

			<Show when={addingRelation()}>
				{(pending) => (
					<Modal title="Add relationship" open={true} onClose={handleCancelAddRelation}>
						<RelationForm
							entities={props.entities}
							source={pending().source}
							target={pending().target}
							onCreated={handleCreatedRelation}
							onCancel={handleCancelAddRelation}
						/>
					</Modal>
				)}
			</Show>
		</>
	);
}
