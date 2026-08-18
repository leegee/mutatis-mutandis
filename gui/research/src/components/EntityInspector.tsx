import { createEffect, createSignal, For, Show } from "solid-js";

import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

import EntityForm from "./EntityForm";
import { useConfirm, usePrompt } from "./Modal";
import { addEntityTag, deleteEntity, removeEntityTag, } from "~/db/repository";

interface EntityInspectorProps {
    entity: Entity | undefined;
    entities: Entity[];
    relations: Relation[];

    onChanged?: (entity: Entity) => void | Promise<void>;
    onClose?: (entity: Entity) => void;
}

export default function EntityInspector(props: EntityInspectorProps) {
    const [editing, setEditing] = createSignal(false);
    const [currentEntity, setCurrentEntity] = createSignal<Entity>(props.entity!);
    const confirm = useConfirm();
    const prompt = usePrompt();

    createEffect(() => {
        if (props.entity) {
            setCurrentEntity(props.entity);
        }
    });

    async function handleAddTag() {
        const entity = currentEntity();
        const tag = await prompt("Tag:");
        if (!tag?.trim()) {
            return;
        }
        const updated = await addEntityTag(entity, tag);
        setCurrentEntity(updated);
        await props.onChanged?.(updated);
    }


    async function handleRemoveTag(tag: string) {
        const entity = currentEntity();
        const updated = await removeEntityTag(entity, tag);
        setCurrentEntity(updated);
        await props.onChanged?.(updated);
    }


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


    async function handleDelete() {
        const entity = props.entity;
        if (!entity) {
            return;
        }

        const ok = await confirm(`Delete "${ entity.label }"?`,);
        if (!ok) {
            return;
        }

        await deleteEntity(entity.id);
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
                                <EntityForm entity={entity()}
                                    onUpdated={async (updated: Entity) => {
                                        setEditing(false);
                                        await props.onChanged?.(updated);
                                    }}
                                    onCancel={() => setEditing(false)}
                                />
                            }
                        >
                            {/* NORMAL INSPECTOR VIEW */}
                            <header class="fixed surface-container-high top-padding" style="top:0">
                                <nav>
                                    <div class="max">
                                        <h2> {entity().label} </h2>
                                        <span> {entity().type} </span>
                                    </div>

                                    <button class="circle transparent"
                                        type="button"
                                        title="Close"
                                        onClick={() => props.onClose?.(entity())}
                                    >
                                        <i>close</i>
                                    </button>
                                </nav>
                            </header>

                            <Show when={entity().description}>
                                <section>
                                    <p> {entity().description} </p>
                                </section>
                            </Show>

                            <Show when={entity().aliases.length > 0} >
                                <section>
                                    <h3>Aliases</h3>
                                    <div class="row wrap">
                                        <For each={entity().aliases} >
                                            {(alias) => (
                                                <span class="chip">
                                                    {alias}
                                                </span>
                                            )}
                                        </For>
                                    </div>
                                </section>
                            </Show>

                            <section>
                                <nav>
                                    <h3 class="max">Tags</h3>

                                    <button type="button" class="small transparent border small circle" onClick={handleAddTag} >
                                        <i class="small">add</i>
                                    </button>
                                </nav>

                                <Show when={entity().tags.length > 0} fallback={<p>No tags.</p>} >
                                    <div class="row wrap">
                                        <For each={entity().tags}>
                                            {(tag) => (
                                                <span class="chip">
                                                    {tag}

                                                    <button type="button"
                                                        class="transparent small"
                                                        title={`Remove ${ tag }`}
                                                        aria-label={`Remove tag ${ tag }`}
                                                        onClick={() => handleRemoveTag(tag)}
                                                    >
                                                        <i class="small">close</i>
                                                    </button>
                                                </span>
                                            )}
                                        </For>
                                    </div>
                                </Show>
                            </section>

                            <section>
                                <h3>Relationships</h3>

                                <Show when={outgoing().length > 0 || incoming().length > 0}
                                    fallback={<p> No relationships yet. </p>}
                                >
                                    <Show when={outgoing().length > 0} >
                                        <h4>Outgoing</h4>

                                        <ul class="list no-space border">
                                            <For each={outgoing()} >
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
                                        <h4>Incoming</h4>

                                        <ul class="list no-space border">
                                            <For each={incoming()} >
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
                                <button type="button" onClick={() => setEditing(true)} >
                                    Edit
                                </button>

                                <button type="button" class="error" onClick={handleDelete} >
                                    Delete
                                </button>
                            </nav>
                        </Show>
                    </aside>
                )}
            </Show>

        </>
    );
}