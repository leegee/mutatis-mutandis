import { createEffect, createSignal, For, Show } from "solid-js";

import type { Entity, EntityType } from "~/domain/entity";
import {
    createEntity,
    updateEntity,
} from "~/db/repository";

import EntityAutocomplete from "./EntityAutocomplete";

const entityTypes: EntityType[] = [
    "concept",
    "lexeme",
    "motif",
    "animal",
    "person",
    "source",
];

interface EntityFormProps {
    entity?: Entity;

    onCreated?: (entity: Entity) => void | Promise<void>;
    onUpdated?: (entity: Entity) => void | Promise<void>;
    onCancel?: () => void;
}

export default function EntityForm(props: EntityFormProps) {
    const editing = () => !!props.entity;

    const [label, setLabel] = createSignal(
        props.entity?.label ?? "",
    );

    const [type, setType] = createSignal<EntityType>(
        props.entity?.type ?? "concept",
    );

    const [selected, setSelected] =
        createSignal<Entity | undefined>(
            props.entity,
        );

    const [saving, setSaving] = createSignal(false);

    createEffect(() => {
        const entity = props.entity;

        if (entity) {
            setLabel(entity.label);
            setType(entity.type);
            setSelected(entity);
        } else {
            setLabel("");
            setType("concept");
            setSelected(undefined);
        }
    });

    function handleInput(value: string) {
        setLabel(value);

        if (!editing()) {
            setSelected(undefined);
        }
    }

    function handleSelect(entity: Entity) {
        setLabel(entity.label);
        setType(entity.type);
        setSelected(entity);
    }

    async function submit(event: SubmitEvent) {
        event.preventDefault();

        if (saving()) {
            return;
        }

        const value = label().trim();

        if (!value) {
            return;
        }

        /*
         * Add mode:
         *
         * If autocomplete selected an existing entity,
         * don't create a duplicate.
         */
        if (!editing() && selected()) {
            return;
        }

        setSaving(true);

        try {
            if (editing() && props.entity) {
                const updated = await updateEntity(
                    props.entity,
                    {
                        label: value,
                        type: type(),
                    },
                );

                await props.onUpdated?.(updated);
            } else {
                const created = await createEntity(
                    value,
                    type(),
                );

                setLabel("");
                setSelected(undefined);

                await props.onCreated?.(created);
            }
        } finally {
            setSaving(false);
        }
    }

    return (
        <form onSubmit={submit}>
            <div class="">
                <div class="field">
                    <Show when={!editing()}
                        fallback={
                            <div class="field label border">
                                <input
                                    value={label()}
                                    disabled={saving()}
                                    onInput={(event) =>
                                        setLabel(
                                            event.currentTarget.value,
                                        )
                                    }
                                />
                                <label>Label</label>
                            </div>
                        }
                    >
                        <EntityAutocomplete
                            value={label()}
                            onInput={handleInput}
                            onSelect={handleSelect}
                            disabled={saving()}
                        />
                    </Show>

                    <Show when={!editing() && selected()}>
                        {(entity) => (
                            <small>
                                Existing {entity().type}:{" "}
                                <strong>
                                    {entity().label}
                                </strong>
                            </small>
                        )}
                    </Show>
                </div>

                <div class="field label border">
                    <select
                        value={type()}
                        disabled={saving()}
                        onChange={(event) =>
                            setType(
                                event.currentTarget.value as EntityType,
                            )
                        }
                    >
                        <For each={entityTypes}>
                            {(entityType) => (
                                <option value={entityType}>
                                    {entityType}
                                </option>
                            )}
                        </For>
                    </select>

                    <label>Type</label>
                </div>

                <nav class="footer">
                    <button
                        type="submit"
                        disabled={
                            !label().trim() ||
                            saving() ||
                            (!editing() && !!selected())
                        }
                    >
                        {saving()
                            ? editing()
                                ? "Saving…"
                                : "Adding…"
                            : editing()
                                ? "Save"
                                : "Add"}
                    </button>

                    <Show when={editing()}>
                        <button
                            type="button"
                            class="transparent"
                            disabled={saving()}
                            onClick={() =>
                                props.onCancel?.()
                            }
                        >
                            Cancel
                        </button>
                    </Show>
                </nav>
            </div>
        </form>
    );
}
