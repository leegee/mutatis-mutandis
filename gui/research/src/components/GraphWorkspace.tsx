import { createSignal } from "solid-js";

import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

import GraphView from "~/components/GraphView";
import EntityInspector from "~/components/EntityInspector";

interface GraphWorkspaceProps {
    entities: Entity[];
    relations: Relation[];
}

export default function GraphWorkspace(
    props: GraphWorkspaceProps,
) {
    const [selected, setSelected] =
        createSignal<Entity>();

    return (
        <div
            style={{
                display: "grid",
                "grid-template-columns":
                    "minmax(0, 1fr) 320px",
                gap: "1rem",
                height: "70vh",
                "min-height": "500px",
            }}
        >
            <div style={{ "min-width": "0" }}>
                <GraphView
                    entities={props.entities}
                    relations={props.relations}
                    onSelectEntity={setSelected}
                />
            </div>

            <div
                class="surface-container"
                style={{
                    "overflow-y": "auto",
                }}
            >
                <EntityInspector
                    entity={selected()}
                    entities={props.entities}
                    relations={props.relations}
                    onClose={() => setSelected(undefined)}
                />
            </div>
        </div>
    );
}
