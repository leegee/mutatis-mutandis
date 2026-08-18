import { createSignal, Show } from "solid-js";

import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

import {
    deleteRelation,
} from "~/db/repository";

import RelationForm from "./RelationForm";

interface RelationInspectorProps {
    relation: Relation | undefined;
    entities: Entity[];

    onChanged?: () => void | Promise<void>;
    onClose?: () => void;
}

export default function RelationInspector(
    props: RelationInspectorProps,
) {
    const [editing, setEditing] = createSignal(false);

    function entityLabel(id: string): string {
        return (
            props.entities.find(
                (entity) => entity.id === id,
            )?.label ?? id
        );
    }

    async function handleDelete() {
        const relation = props.relation;

        if (!relation) {
            return;
        }

        const confirmed = window.confirm(
            "Delete this relationship?",
        );

        if (!confirmed) {
            return;
        }

        await deleteRelation(relation.id);

        await props.onChanged?.();
        props.onClose?.();
    }

    return (
        <aside class="padding">
            <Show
                when={props.relation}
                fallback={
                    <div>
                        <h3>No relationship selected</h3>
                        <p>
                            Select a relationship in the
                            graph.
                        </p>
                    </div>
                }
            >
                {(relation) => (
                    <Show
                        when={!editing()}
                        fallback={
                            <RelationForm
                                relation={relation()}
                                entities={props.entities}
                                onUpdated={async () => {
                                    setEditing(false);
                                    await props.onChanged?.();
                                }}
                                onCancel={() => setEditing(false)}
                            />
                        }
                    >
                        <header>
                            <nav>
                                <h3>Relationship</h3>

                                <button
                                    class="circle transparent"
                                    type="button"
                                    onClick={() =>
                                        props.onClose?.()
                                    }
                                >
                                    ×
                                </button>
                            </nav>
                        </header>

                        <div class="padding">
                            <p>
                                <strong>
                                    {entityLabel(
                                        relation().sourceId,
                                    )}
                                </strong>

                                {" → "}

                                <strong>
                                    {relation().type}
                                </strong>

                                {" → "}

                                <strong>
                                    {entityLabel(
                                        relation().targetId,
                                    )}
                                </strong>
                            </p>

                            <div class="row">
                                <button type="button" onClick={() => setEditing(true)} >
                                    Edit
                                </button>

                                <button type="button" class="error" onClick={handleDelete} >
                                    Delete
                                </button>
                            </div>
                        </div>
                    </Show>
                )}
            </Show>
        </aside>
    );
}
