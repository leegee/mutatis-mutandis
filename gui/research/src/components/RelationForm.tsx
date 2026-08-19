import { createEffect, createSignal, For, Show } from "solid-js";

import type { Entity } from "~/domain/entity";
import type { Relation, RelationType } from "~/domain/relation";

import {
    createRelation,
    updateRelation,
} from "~/db/respository";

import EntityAutocomplete from "~/components/EntityAutocomplete";

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
    relation?: Relation;

    entities?: Entity[];

    source?: Entity;
    target?: Entity;

    onCreated?: (relation: Relation) => void | Promise<void>;
    onUpdated?: (relation: Relation) => void | Promise<void>;
    onCancel?: () => void;
}

export default function RelationForm(
    props: RelationFormProps,
) {
    const editing = () => !!props.relation;

    const [source, setSource] =
        createSignal<Entity>();

    const [target, setTarget] =
        createSignal<Entity>();

    const [relationType, setRelationType] =
        createSignal<RelationType>("related-to");

    const [sourceValue, setSourceValue] =
        createSignal("");

    const [targetValue, setTargetValue] =
        createSignal("");

    const [saving, setSaving] =
        createSignal(false);

    // When editing, initialise the form from the existing relation.
    createEffect(() => {
        const relation = props.relation;

        if (relation) {
            if (!props.entities) return;
            const sourceEntity = props.entities.find(
                (entity) => entity.id === relation.sourceId,
            );

            const targetEntity = props.entities.find(
                (entity) => entity.id === relation.targetId,
            );

            setSource(sourceEntity);
            setTarget(targetEntity);

            setSourceValue(sourceEntity?.label ?? "");
            setTargetValue(targetEntity?.label ?? "");

            setRelationType(relation.type);
            return;
        }

        setSource(props.source);
        setTarget(props.target);

        setSourceValue(props.source?.label ?? "");
        setTargetValue(props.target?.label ?? "");

        setRelationType("related-to");
    });

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
            if (props.relation) {
                const relation = await updateRelation(
                    props.relation,
                    {
                        sourceId: sourceEntity.id,
                        type: relationType(),
                        targetId: targetEntity.id,
                    },
                );

                await props.onUpdated?.(relation);
            } else {
                const relation = await createRelation(
                    sourceEntity.id,
                    relationType(),
                    targetEntity.id,
                );

                setSource(undefined);
                setTarget(undefined);

                setSourceValue("");
                setTargetValue("");

                await props.onCreated?.(relation);
            }
        } finally {
            setSaving(false);
        }
    }

    return (
        <form onSubmit={submit}>
            <div class="">
                <h3>
                    {editing()
                        ? "Edit Relationship"
                        : "Add Relationship"}
                </h3>
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
                                event.currentTarget
                                    .value as RelationType,
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

                    <label> Relationship </label>
                </div>

                <div class="field">
                    <EntityAutocomplete
                        value={targetValue()}
                        onInput={inputTarget}
                        onSelect={selectTarget}
                        disabled={saving()}
                        placeholder="Target Entity"
                    />
                </div>

                <nav class="footer">
                    <button type="submit" disabled={saving() || !source() || !target()} >
                        {saving()
                            ? editing()
                                ? "Saving…"
                                : "Adding…"
                            : editing()
                                ? "Save relationship"
                                : "Add relationship"}
                    </button>

                    <Show when={editing()}>
                        <button
                            type="button"
                            class="transparent"
                            disabled={saving()}
                            onClick={() => props.onCancel?.()}
                        >
                            Cancel
                        </button>
                    </Show>
                </nav>
            </div >
        </form >
    );
}
