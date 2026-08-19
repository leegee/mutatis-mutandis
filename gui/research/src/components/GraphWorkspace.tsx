import { createSignal, Show } from "solid-js";

import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

import {
    createEntity,
    deleteEntity,
    deleteRelation,
} from "~/db/respository";

import GraphView from "~/components/GraphView";
import EntityInspector from "~/components/EntityInspector";
import RelationInspector from "~/components/RelationInspector";
import { Modal } from "~/components/Modal";
import RelationForm from "~/components/RelationForm";
import { usePrompt } from "./Modal";

interface GraphWorkspaceProps {
    entities: Entity[];
    relations: Relation[];
}

export default function GraphWorkspace(props: GraphWorkspaceProps,) {
    const [selectedEntity, setSelectedEntity] = createSignal<Entity>();
    const [selectedRelation, setSelectedRelation] = createSignal<Relation>();
    const [addingRelation, setAddingRelation] = createSignal<{
        source: Entity;
        target: Entity;
    }>();

    const prompt = usePrompt();

    async function handleAddEntity() {
        const label = await prompt("Node label");
        if (!label?.trim()) return;
        const entity = await createEntity(label.trim());
        setSelectedEntity(entity);
        setSelectedRelation(undefined);
    }

    function handleEditEntity(entity: Entity) {
        setSelectedEntity(entity);
        setSelectedRelation(undefined);
    }

    async function handleDeleteEntity(entity: Entity) {
        await deleteEntity(entity.id);

        if (selectedEntity()?.id === entity.id) {
            setSelectedEntity(undefined);
        }
    }

    function handleAddRelation(
        sourceId: string,
        targetId: string,
    ) {
        const source = props.entities.find((entity) => entity.id === sourceId,);
        const target = props.entities.find((entity) => entity.id === targetId,);

        if (!source || !target) {
            return;
        }

        setAddingRelation({ source, target, });
    }

    function handleCreatedRelation(
        relation: Relation,
    ) {
        setAddingRelation(undefined);
        setSelectedRelation(relation);
        setSelectedEntity(undefined);
    }

    function handleCancelAddRelation() {
        setAddingRelation(undefined);
    }

    function handleEditRelation(relation: Relation) {
        setSelectedRelation(relation);
        setSelectedEntity(undefined);
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
            <div class="background"
                style={{
                    display: "grid",
                    "grid-template-columns": selectedEntity() || selectedRelation()
                        ? "minmax(0, 1fr) 30vw"
                        : "minmax(0, 1fr)",
                    gap: "1rem",
                    height: "100vh",
                    "min-height": "500px",
                }}
            >
                <div style={{ "min-width": "0", }}
                >
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

                <Show when={selectedEntity() || selectedRelation()} >
                    <div class="surface-container-high medium-elevation left-padding right-padding"
                        style={{ "overflow-y": "auto", }}
                    >
                        <Show when={selectedEntity()}
                            fallback={
                                <RelationInspector
                                    relation={selectedRelation()}
                                    entities={props.entities}
                                    onClose={() => setSelectedRelation(undefined)}
                                />
                            }
                        >
                            {(entity) => (
                                <EntityInspector
                                    entity={entity()}
                                    entities={props.entities}
                                    relations={props.relations}
                                    onChanged={handleEntityChanged}
                                    onClose={() => setSelectedEntity(undefined)}
                                />
                            )}
                        </Show>
                    </div>
                </Show>
            </div>

            <Show when={addingRelation()}>
                {(pending) => (
                    <Modal title="Add relationship"
                        open={true}
                        onClose={
                            handleCancelAddRelation
                        }
                    >
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