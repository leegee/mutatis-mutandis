import { createSignal, For } from "solid-js";

import type { Entity } from "~/domain/entity";
import { createRelation, listEntities } from "~/db/repository";

import EntityAutocomplete from "~/components/EntityAutocomplete";
import { RelationType } from "~/domain/relation";

const relationTypes: RelationType[] = [
    "related-to",
    "contrasts-with",
    "describes",
    "expresses",
    "attested-in",
    "supports",
    "possibly-derived-from",
];

interface RelationFormProps {
    onCreated?: () => void | Promise<void>;
}

export default function RelationForm(props: RelationFormProps) {
    const [source, setSource] = createSignal<Entity>();
    const [target, setTarget] = createSignal<Entity>();

    const [relationType, setRelationType] =
        createSignal<RelationType>("related-to");

    const [sourceValue, setSourceValue] = createSignal("");
    const [targetValue, setTargetValue] = createSignal("");

    const [saving, setSaving] = createSignal(false);

    function selectSource(entity: Entity) {
        setSource(entity);
        setSourceValue(entity.label);
    }

    function selectTarget(entity: Entity) {
        setTarget(entity);
        setTargetValue(entity.label);
    }

    function inputSource(value: string) {
        setSourceValue(value);
        setSource(undefined);
    }

    function inputTarget(value: string) {
        setTargetValue(value);
        setTarget(undefined);
    }

    async function submit(event: SubmitEvent) {
        event.preventDefault();

        const sourceEntity = source();
        const targetEntity = target();

        if (
            saving() ||
            !sourceEntity ||
            !targetEntity
        ) {
            return;
        }

        setSaving(true);

        try {
            await createRelation(
                sourceEntity.id,
                relationType(),
                targetEntity.id,
            );

            setSource(undefined);
            setTarget(undefined);

            setSourceValue("");
            setTargetValue("");

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
                        value={sourceValue()}
                        onInput={inputSource}
                        onSelect={selectSource}
                        disabled={saving()}
                        placeholder="Source entity"
                    />
                </div>

                <div class="field label border">
                    <select
                        value={relationType()}
                        onChange={(event) =>
                            setRelationType(
                                event.currentTarget.value as RelationType,
                            )
                        }
                        disabled={saving()}
                    >
                        <For each={relationTypes}>
                            {(type) => (
                                <option value={type}>
                                    {type}
                                </option>
                            )}
                        </For>
                    </select>

                    <label>Relationship</label>
                </div>

                <div class="field">
                    <EntityAutocomplete
                        value={targetValue()}
                        onInput={inputTarget}
                        onSelect={selectTarget}
                        disabled={saving()}
                        placeholder="Target entity"
                    />
                </div>

                <div class="field">
                    <button
                        type="submit"
                        disabled={
                            saving() ||
                            !source() ||
                            !target()
                        }
                    >
                        {saving()
                            ? "Adding…"
                            : "Add relationship"}
                    </button>
                </div>
            </div>
        </form>
    );
}
