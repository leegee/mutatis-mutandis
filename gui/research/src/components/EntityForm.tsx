import { createSignal, For, Show } from "solid-js";

import type { Entity, EntityType } from "~/domain/entity";
import { createEntity } from "~/db/repository";
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
    onCreated?: () => void | Promise<void>;
}

export default function EntityForm(props: EntityFormProps) {
    const [label, setLabel] = createSignal("");
    const [type, setType] = createSignal<EntityType>("concept");
    const [selected, setSelected] = createSignal<Entity | undefined>();
    const [saving, setSaving] = createSignal(false);

    function handleInput(value: string) {
        setLabel(value);
        setSelected(undefined);
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

        // An existing entity has been selected.
        // Don't create a duplicate.
        if (selected()) {
            return;
        }

        setSaving(true);

        try {
            await createEntity(value, type());

            setLabel("");
            setSelected(undefined);

            await props.onCreated?.();
        } finally {
            setSaving(false);
        }
    }

    return (
        <form onSubmit={submit}>
            <div class="row">
                <div class="field">
                    <EntityAutocomplete
                        value={label()}
                        onInput={handleInput}
                        onSelect={handleSelect}
                        disabled={saving()}
                        placeholder="e.g. hvítr"
                    />

                    <Show when={selected()}>
                        {(entity) => (
                            <small>
                                Existing {entity().type}:{" "}
                                <strong>{entity().label}</strong>
                            </small>
                        )}
                    </Show>
                </div>

                <div class="field label border">
                    <select
                        value={type()}
                        disabled={saving() || !!selected()}
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

                <div class="field">
                    <button
                        type="submit"
                        disabled={
                            !label().trim() ||
                            saving() ||
                            !!selected()
                        }
                    >
                        {saving() ? "Adding…" : "Add"}
                    </button>
                </div>
            </div>
        </form>
    );
}
