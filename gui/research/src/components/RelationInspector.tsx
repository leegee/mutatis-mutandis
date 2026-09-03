import { createEffect, createSignal, Show } from "solid-js";
import { deleteRelation } from "~/db/respository";
import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

import RelationForm from "./RelationForm";

interface RelationInspectorProps {
    relation: Relation | undefined;
    entities: Entity[];
    editing?: boolean;
    onChanged?: (relation: Relation) => void | Promise<void>;
    onClose?: (relation: Relation) => void;
}

export default function RelationInspector(props: RelationInspectorProps) {
    const [editing, setEditing] = createSignal(props.editing);

    createEffect(() => {
        if (props.editing !== undefined) {
            setEditing(props.editing);
        }
    });

    function entityLabel(id: string): string {
        return props.entities.find((entity) => entity.id === id)?.label ?? id;
    }

    async function handleDelete() {
        const relation = props.relation;
        if (!relation) return;

        const confirmed = window.confirm("Delete this relationship?");
        if (!confirmed) return;

        await deleteRelation(relation.id);

        await props.onChanged?.(relation);
        props.onClose?.(relation);
    }

    return (
        <Show when={props.relation}>
            {(relation) => (
                <aside class="padding surface-container">
                    <Show
                        when={!editing()}
                        fallback={
                            <RelationForm
                                relation={relation()}
                                entities={props.entities}
                                onUpdated={async (updated) => {
                                    setEditing(false);
                                    await props.onChanged?.(updated);
                                }}
                                onCancel={() => setEditing(false)}
                            />
                        }
                    >
                        <header class="bottom-margin">
                            <nav>
                                <button
                                    class="circle transparent"
                                    type="button"
                                    title="Close"
                                    onClick={() => props.onClose?.(relation())}
                                >
                                    <i>close</i>
                                </button>

                                <h2 class="max"> Relationship </h2>
                            </nav>
                        </header>

                        <section class="surface-container">
                            <p class="padding">
                                <strong> {entityLabel(relation().sourceId)} </strong>
                                {" → "}
                                <strong> {relation().type} </strong>
                                {" → "}
                                <strong>{entityLabel(relation().targetId)}</strong>
                            </p>

                            <nav class="footer">
                                <button type="button" class="error" onClick={handleDelete}>
                                    Delete
                                </button>
                                <button type="button" onClick={() => setEditing(true)}>
                                    Edit
                                </button>
                            </nav>
                        </section>
                    </Show>
                </aside>
            )}
        </Show>
    );
}
