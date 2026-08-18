import { createSignal, Show } from "solid-js";

import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

import {
    createEntity,
    createRelation,
    deleteEntity,
    deleteRelation,
} from "~/db/repository";

import GraphView from "~/components/GraphView";
import EntityInspector from "~/components/EntityInspector";
import RelationInspector from "~/components/RelationInspector";
import Prompt from "~/components/Modal/Prompt";

interface GraphWorkspaceProps {
    entities: Entity[];
    relations: Relation[];
}

export default function GraphWorkspace(
    props: GraphWorkspaceProps,
) {
    const [selectedEntity, setSelectedEntity] =
        createSignal<Entity>();

    const [selectedRelation, setSelectedRelation] =
        createSignal<Relation>();

    const [addingEntity, setAddingEntity] =
        createSignal(false);

    async function handleAddEntity() {
        setAddingEntity(true);
    }

    async function handleConfirmAddEntity(
        label: string,
    ) {
        const entity = await createEntity(label);

        setAddingEntity(false);
        setSelectedEntity(entity);
        setSelectedRelation(undefined);
    }

    function handleCancelAddEntity() {
        setAddingEntity(false);
    }

    async function handleEditEntity(
        entity: Entity,
    ) {
        setSelectedEntity(entity);
        setSelectedRelation(undefined);
    }

    async function handleDeleteEntity(
        entity: Entity,
    ) {
        await deleteEntity(entity.id);

        if (selectedEntity()?.id === entity.id) {
            setSelectedEntity(undefined);
        }
    }

    async function handleAddRelation(
        sourceId: string,
        targetId: string,
    ) {
        const type = window.prompt(
            "Relationship type:",
            "related-to",
        );

        if (!type?.trim()) {
            return;
        }

        const relation = await createRelation(
            sourceId,
            type.trim() as any,
            targetId,
        );

        setSelectedRelation(relation);
        setSelectedEntity(undefined);
    }

    async function handleEditRelation(
        relation: Relation,
    ) {
        setSelectedRelation(relation);
        setSelectedEntity(undefined);
    }

    async function handleDeleteRelation(
        relation: Relation,
    ) {
        await deleteRelation(relation.id);

        if (
            selectedRelation()?.id === relation.id
        ) {
            setSelectedRelation(undefined);
        }
    }

    return (
        <>
            <div
                style={{
                    display: "grid",
                    "grid-template-columns":
                        selectedEntity() ||
                            selectedRelation()
                            ? "minmax(0, 1fr) 30vw"
                            : "minmax(0, 1fr)",
                    gap: "1rem",
                    height: "90vh",
                    "min-height": "500px",
                }}
            >
                <div style={{ "min-width": "0" }}>
                    <GraphView
                        entities={props.entities}
                        relations={props.relations}

                        onSelectEntity={(entity) => {
                            setSelectedEntity(entity);
                            setSelectedRelation(undefined);
                        }}

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

                <Show
                    when={
                        selectedEntity() ||
                        selectedRelation()
                    }
                >
                    <div
                        class="surface-container-high medium-elevation left-padding right-padding"
                        style={{
                            "overflow-y": "auto",
                        }}
                    >
                        <Show
                            when={selectedEntity()}
                            fallback={
                                <RelationInspector
                                    relation={selectedRelation()}
                                    entities={props.entities}
                                    onClose={() =>
                                        setSelectedRelation(undefined)
                                    }
                                />
                            }
                        >
                            {(entity) => (
                                <EntityInspector
                                    entity={entity()}
                                    entities={props.entities}
                                    relations={props.relations}
                                    onClose={() =>
                                        setSelectedEntity(undefined)
                                    }
                                />
                            )}
                        </Show>
                    </div>
                </Show>
            </div>

            <Prompt
                open={addingEntity()}
                title="Add node"
                placeholder="Node label"
                confirmLabel="Add"
                cancelLabel="Cancel"
                onConfirm={handleConfirmAddEntity}
                onCancel={handleCancelAddEntity}
            />
        </>
    );
}
