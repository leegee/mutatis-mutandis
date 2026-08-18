import { createSignal, Show } from "solid-js";

import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

import GraphView from "~/components/GraphView";
import EntityInspector from "~/components/EntityInspector";
import RelationInspector from "~/components/RelationInspector";

interface GraphWorkspaceProps {
    entities: Entity[];
    relations: Relation[];
}

export default function GraphWorkspace(
    props: GraphWorkspaceProps,
) {
    const [selectedEntity, setSelectedEntity] = createSignal<Entity>();
    const [selectedRelation, setSelectedRelation] = createSignal<Relation>();

    return (
        <div
            style={{
                display: "grid",
                "grid-template-columns":
                    selectedEntity() || selectedRelation()
                        ? "minmax(0, 1fr) 320px"
                        : "minmax(0, 1fr)",
                gap: "1rem",
                height: "70vh",
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
                />
            </div>

            <Show when={selectedEntity() || selectedRelation()}>
                <div
                    class="surface-container-high medium-elevation left-padding right-padding"
                    style={{ "overflow-y": "auto" }}
                >
                    <Show
                        when={selectedEntity()}
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
                                onClose={() => setSelectedEntity(undefined)}
                            />
                        )}
                    </Show>
                </div>
            </Show>
        </div>
    );
}
