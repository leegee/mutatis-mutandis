import { createSignal, For, Show } from "solid-js";

import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

import { deleteEntity } from "~/db/repository";
import EntityForm from "./EntityForm";
import Confirm from "./Modal/Confirm";

interface EntityInspectorProps {
    entity: Entity | undefined;
    entities: Entity[];
    relations: Relation[];

    onChanged?: (entity: Entity) => void | Promise<void>;
    onClose?: (entity: Entity) => void;
}

export default function EntityInspector(
    props: EntityInspectorProps,
) {
    const [editing, setEditing] = createSignal(false);
    const [confirmDelete, setConfirmDelete] = createSignal(false);

    function entityLabel(id: string): string {
        return (props.entities.find((entity) => entity.id === id,)?.label ?? id);
    }

    function outgoing(): Relation[] {
        const entity = props.entity;

        if (!entity) {
            return [];
        }

        return props.relations.filter((relation) => relation.sourceId === entity.id,);
    }

    function incoming(): Relation[] {
        const entity = props.entity;

        if (!entity) {
            return [];
        }

        return props.relations.filter((relation) => relation.targetId === entity.id,);
    }

    function handleDelete() {
        if (!props.entity) {
            return;
        }

        setConfirmDelete(true);
    }

    function handleCancelDelete() {
        setConfirmDelete(false);
    }

    async function handleConfirmDelete() {
        const entity = props.entity;

        if (!entity) {
            setConfirmDelete(false);
            return;
        }

        await deleteEntity(entity.id);

        setConfirmDelete(false);
        setEditing(false);

        await props.onChanged?.(entity);
        props.onClose?.(entity);
    }

    return (
        <>
            <Show when={props.entity} fallback={<></>}>
                {(entity) => (
                    <aside class="padding">
                        <Show when={!editing()}
                            fallback={
                                <EntityForm
                                    entity={entity()}
                                    onUpdated={async (updated: Entity) => {
                                        setEditing(false);
                                        await props.onChanged?.(updated);
                                    }}
                                    onCancel={() =>
                                        setEditing(false)
                                    }
                                />
                            }
                        >
                            {/* NORMAL INSPECTOR VIEW */}
                            <header>
                                <nav>
                                    <div class="max">
                                        <h2>
                                            {entity().label}
                                        </h2>

                                        <small>
                                            {entity().type}
                                        </small>
                                    </div>

                                    <button
                                        class="circle transparent"
                                        type="button"
                                        title="Close"
                                        onClick={() =>
                                            props.onClose?.(entity())
                                        }
                                    >
                                        <i>close</i>
                                    </button>
                                </nav>
                            </header>

                            <Show when={entity().description}>
                                <p>
                                    {entity().description}
                                </p>
                            </Show>

                            <Show when={entity().aliases.length > 0} >
                                <section>
                                    <h5>Aliases</h5>

                                    <For
                                        each={entity().aliases}
                                    >
                                        {(alias) => (
                                            <span class="chip">
                                                {alias}
                                            </span>
                                        )}
                                    </For>
                                </section>
                            </Show>

                            <Show when={entity().tags.length > 0} >
                                <section>
                                    <h5>Tags</h5>

                                    <div class="row wrap">
                                        <For
                                            each={entity().tags}
                                        >
                                            {(tag) => (
                                                <span class="chip">
                                                    {tag}
                                                </span>
                                            )}
                                        </For>
                                    </div>
                                </section>
                            </Show>

                            <section>
                                <h5>Relationships</h5>

                                <Show when={outgoing().length > 0 || incoming().length > 0}
                                    fallback={
                                        <p>
                                            No relationships yet.
                                        </p>
                                    }
                                >
                                    <Show when={outgoing().length > 0} >
                                        <h6>Outgoing</h6>

                                        <ul>
                                            <For
                                                each={outgoing()}
                                            >
                                                {(relation) => (
                                                    <li>
                                                        {relation.type}
                                                        {" → "}
                                                        {
                                                            entityLabel(
                                                                relation.targetId,
                                                            )
                                                        }
                                                    </li>
                                                )}
                                            </For>
                                        </ul>
                                    </Show>

                                    <Show when={incoming().length > 0} >
                                        <h6>Incoming</h6>

                                        <ul>
                                            <For
                                                each={incoming()}
                                            >
                                                {(relation) => (
                                                    <li>
                                                        {relation.type}
                                                        {" → "}
                                                        {
                                                            entityLabel(
                                                                relation.sourceId,
                                                            )
                                                        }
                                                    </li>
                                                )}
                                            </For>
                                        </ul>
                                    </Show>
                                </Show>
                            </section>

                            <nav class="footer">
                                <button
                                    type="button"
                                    onClick={() =>
                                        setEditing(true)
                                    }
                                >
                                    Edit
                                </button>

                                <button
                                    type="button"
                                    class="error"
                                    onClick={handleDelete}
                                >
                                    Delete
                                </button>
                            </nav>
                        </Show>
                    </aside>
                )}
            </Show>

            <Confirm
                open={confirmDelete()}
                title="Delete entity"
                message={
                    props.entity
                        ? `Delete "${ props.entity.label }" ? Its relationships will also be removed.`
                        : ""
                }
                confirmLabel="Delete"
                cancelLabel="Cancel"
                onConfirm={handleConfirmDelete}
                onCancel={handleCancelDelete}
            />
        </>
    );
}