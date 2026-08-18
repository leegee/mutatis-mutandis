import { createSignal, Show } from "solid-js";

import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

import { deleteRelation, } from "~/db/repository";

import RelationForm from "./RelationForm";

interface RelationInspectorProps {
    relation: Relation | undefined;
    entities: Entity[];
    onChanged?: (relation: Relation) => void | Promise<void>;
    onClose?: (relation: Relation) => void;
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

        await props.onChanged?.(relation);
        props.onClose?.(relation);
    }

    return (
        <Show when={props.relation} >
            {(relation) => (
                <aside>
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
                        <header>
                            <nav>
                                <h3 class="max">Relationship</h3>

                                <button
                                    class="circle transparent"
                                    type="button"
                                    onClick={() => props.onClose?.(relation())}
                                >
                                    <i>close</i>
                                </button>
                            </nav>
                        </header>

                        <div class="padding">
                            <p>
                                <strong> {entityLabel(relation().sourceId,)} </strong>

                                {" → "}

                                <strong> {relation().type} </strong>

                                {" → "}

                                <strong>
                                    {entityLabel(
                                        relation().targetId,
                                    )}
                                </strong>
                            </p>

                            <nav class="footer">
                                <button type="button" onClick={() => setEditing(true)} >
                                    Edit
                                </button>

                                <button type="button" class="error" onClick={handleDelete} >
                                    Delete
                                </button>
                            </nav>
                        </div>
                    </Show>
                </aside>
            )}
        </Show>
    );
}
