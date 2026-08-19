import { createEffect, createSignal, For, Show } from "solid-js";
import {
    addEntityAlias,
    addEntityTag,
    deleteEntity,
    listTags,
    removeEntityAlias,
    removeEntityTag,
} from "~/db/respository";
import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";
import AutoComplete from "./AutoComplete";
import EntityForm from "./EntityForm";
import { useConfirm, usePrompt } from "./Modal";

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
    const [tagInput, setTagInput] = createSignal("");
    const [tags, setTags] = createSignal<string[]>([]);

    createEffect(() => {
        listTags().then(setTags);
    });

    const confirm = useConfirm();
    const prompt = usePrompt();

    createEffect(() => {
        if (props.entity) {
            setCurrentEntity(props.entity);
        }
    });

    async function handleAddTag(tag: string) {
        const value = tag.trim();

        if (!value) {
            return;
        }

        const entity = currentEntity();

        const updated = await addEntityTag(entity, value);

        setCurrentEntity(updated);
        setTagInput("");

        // Refresh the global tag list in case this was a new tag.
        setTags(await listTags());

        await props.onChanged?.(updated);
    }

    async function handleRemoveTag(tag: string) {
        const entity = currentEntity();
        const updated = await removeEntityTag(entity, tag);
        setCurrentEntity(updated);
        await props.onChanged?.(updated);
    }

    async function handleAddAlias() {
        const entity = currentEntity();
        const alias = await prompt("Alias:");
        if (!alias?.trim()) {
            return;
        }

        const updated = await addEntityAlias(entity, alias);
        setCurrentEntity(updated);
        await props.onChanged?.(updated);
    }

    async function handleRemoveAlias(alias: string) {
        const entity = currentEntity();
        const updated = await removeEntityAlias(entity, alias);
        setCurrentEntity(updated);
        await props.onChanged?.(updated);
    }

    function entityLabel(id: string): string {
        return props.entities.find((entity) => entity.id === id)?.label ?? id;
    }

    function outgoing(): Relation[] {
        const entity = props.entity;

        if (!entity) {
            return [];
        }

        return props.relations.filter(
            (relation) => relation.sourceId === entity.id,
        );
    }

    function incoming(): Relation[] {
        const entity = props.entity;

        if (!entity) {
            return [];
        }

        return props.relations.filter(
            (relation) => relation.targetId === entity.id,
        );
    }

    async function handleDelete() {
        const entity = props.entity;
        if (!entity) {
            return;
        }

        const ok = await confirm(`Delete "${ entity.label }"?`);
        if (!ok) {
            return;
        }

        await deleteEntity(entity.id);
        await props.onChanged?.(entity);
        props.onClose?.(entity);
    }

    return (
        <Show when={props.entity} fallback={""}>
            {(entity) => (
                <aside class="padding">
                    <Show
                        when={!editing()}
                        fallback={
                            <EntityForm
                                entity={entity()}
                                onUpdated={async (updated: Entity) => {
                                    setEditing(false);
                                    await props.onChanged?.(updated);
                                }}
                                onCancel={() => setEditing(false)}
                            />
                        }
                    >
                        {/* NORMAL INSPECTOR VIEW */}
                        <header
                            class="fixed surface-container-high top-padding"
                            style="top:0"
                        >
                            <nav>
                                <div class="max">
                                    <h2> {entity().label} </h2>
                                    <span> {entity().type} </span>
                                </div>

                                <button
                                    class="circle transparent"
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

                        <section>
                            <nav>
                                <h3 class="max">Aliases</h3>

                                <button
                                    type="button"
                                    class="small transparent border small circle"
                                    onClick={handleAddAlias}
                                >
                                    <i class="small">add</i>
                                </button>
                            </nav>

                            <Show
                                when={currentEntity().aliases.length > 0}
                                fallback={<p>No aliases.</p>}
                            >
                                <div class="row wrap tiny-space">
                                    <For each={currentEntity().aliases}>
                                        {(alias) => (
                                            <span class="chip small left-padding">
                                                {alias}

                                                <button
                                                    type="button"
                                                    class="transparent small circle no-padding"
                                                    title={`Remove ${ alias }`}
                                                    aria-label={`Remove alias ${ alias }`}
                                                    onClick={() => handleRemoveAlias(alias)}
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
                            <AutoComplete<string>
                                value={tagInput()}
                                items={tags()}
                                getLabel={(tag) => tag}
                                onInput={setTagInput}
                                onSelect={handleAddTag}
                                placeholder="Tags"
                                isTitle
                            />

                            <Show
                                when={currentEntity().tags.length > 0}
                                fallback={<p>No tags.</p>}
                            >
                                <div class="row wrap tiny-space">
                                    <For each={currentEntity().tags}>
                                        {(tag) => (
                                            <span class="small chip left-padding">
                                                {tag}

                                                <button type="button"
                                                    class="transparent small circle no-padding"
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

                            <Show
                                when={outgoing().length > 0 || incoming().length > 0}
                                fallback={<p> No relationships yet. </p>}
                            >
                                <Show when={outgoing().length > 0}>
                                    <h4>Outgoing</h4>

                                    <ul class="list no-space border">
                                        <For each={outgoing()}>
                                            {(relation) => (
                                                <li>
                                                    {relation.type}
                                                    {" → "}
                                                    {entityLabel(relation.targetId)}
                                                </li>
                                            )}
                                        </For>
                                    </ul>
                                </Show>

                                <Show when={incoming().length > 0}>
                                    <h4>Incoming</h4>

                                    <ul class="list no-space border">
                                        <For each={incoming()}>
                                            {(relation) => (
                                                <li>
                                                    {relation.type}
                                                    {" → "}
                                                    {entityLabel(relation.sourceId)}
                                                </li>
                                            )}
                                        </For>
                                    </ul>
                                </Show>
                            </Show>
                        </section>

                        <nav class="footer">
                            <button type="button" onClick={() => setEditing(true)}>
                                Edit
                            </button>

                            <button type="button" class="error" onClick={handleDelete}>
                                Delete
                            </button>
                        </nav>
                    </Show>
                </aside>
            )}
        </Show>
    );
}
