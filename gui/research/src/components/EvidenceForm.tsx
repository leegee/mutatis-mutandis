import { createSignal, For, Show } from "solid-js";
import { createEvidence, updateEvidence } from "~/db/respository";
import type { Entity } from "~/domain/entity";
import type { Evidence, EvidenceStatus } from "~/domain/evidence";
import type { Relation } from "~/domain/relation";

import AutoComplete from "./AutoComplete";

const evidenceStatuses: EvidenceStatus[] = [
    "primary",
    "secondary",
    "interpretive",
    "speculative",
];

interface EvidenceFormProps {
    evidence?: Evidence;

    entities: Entity[];
    relations: Relation[];

    onCreated?: (evidence: Evidence) => void | Promise<void>;
    onUpdated?: (evidence: Evidence) => void | Promise<void>;
    onCancel?: () => void;
}

export default function EvidenceForm(props: EvidenceFormProps) {
    const editing = () => !!props.evidence;

    const [sourceId, setSourceId] = createSignal(props.evidence?.sourceId ?? "");

    const [entityIds, setEntityIds] = createSignal<string[]>(
        props.evidence?.entityIds ?? [],
    );

    const [relationIds, setRelationIds] = createSignal<string[]>(
        props.evidence?.relationIds ?? [],
    );

    const [quote, setQuote] = createSignal(props.evidence?.quote ?? "");

    const [observation, setObservation] = createSignal(
        props.evidence?.observation ?? "",
    );

    const [status, setStatus] = createSignal<EvidenceStatus>(
        props.evidence?.status ?? "primary",
    );

    const [notes, setNotes] = createSignal(props.evidence?.notes ?? "");

    const [saving, setSaving] = createSignal(false);

    const sourceEntities = () =>
        props.entities.filter((entity) => entity.type === "source");

    const selectedEntities = () =>
        props.entities.filter((entity) => entityIds().includes(entity.id));

    const selectedRelations = () =>
        props.relations.filter((relation) => relationIds().includes(relation.id));

    function selectSource(entity: Entity) {
        setSourceId(entity.id);
    }

    function addEntity(entity: Entity) {
        if (entityIds().includes(entity.id)) {
            return;
        }

        setEntityIds([...entityIds(), entity.id]);
    }

    function removeEntity(entityId: string) {
        setEntityIds(entityIds().filter((id) => id !== entityId));
    }

    function addRelation(relation: Relation) {
        if (relationIds().includes(relation.id)) {
            return;
        }

        setRelationIds([...relationIds(), relation.id]);
    }

    function removeRelation(relationId: string) {
        setRelationIds(relationIds().filter((id) => id !== relationId));
    }

    async function submit(event: SubmitEvent) {
        event.preventDefault();

        if (saving()) {
            return;
        }

        if (!sourceId() || !observation().trim()) {
            return;
        }

        setSaving(true);

        try {
            if (editing() && props.evidence) {
                const updated = await updateEvidence(props.evidence, {
                    sourceId: sourceId(),
                    entityIds: entityIds(),
                    relationIds: relationIds(),
                    quote: quote().trim() || undefined,
                    observation: observation().trim(),
                    status: status(),
                    notes: notes().trim() || undefined,
                });

                await props.onUpdated?.(updated);
            } else {
                const created = await createEvidence(
                    sourceId(),
                    observation().trim(),
                    status(),
                    entityIds(),
                    relationIds(),
                    quote().trim() || undefined,
                    notes().trim() || undefined,
                );

                await props.onCreated?.(created);
            }
        } finally {
            setSaving(false);
        }
    }

    return (
        <form onSubmit={submit}>
            <AutoComplete<Entity>
                value={
                    sourceEntities().find((entity) => entity.id === sourceId())?.label ??
                    ""
                }
                items={sourceEntities()}
                getLabel={(entity) => entity.label}
                onInput={() => {
                    // Selection determines the source.
                }}
                onSelect={selectSource}
                placeholder="Source"
                isTitle
            />

            <section>
                <h4>Entities</h4>

                <AutoComplete<Entity>
                    value=""
                    items={props.entities}
                    getLabel={(entity) => entity.label}
                    onInput={() => { }}
                    onSelect={addEntity}
                    placeholder="Add entity"
                />

                <div class="row wrap tiny-space">
                    <For each={selectedEntities()}>
                        {(entity) => (
                            <span class="small chip left-padding">
                                {entity.label}

                                <button
                                    type="button"
                                    class="transparent small circle no-padding"
                                    onClick={() => removeEntity(entity.id)}
                                >
                                    <i class="small">close</i>
                                </button>
                            </span>
                        )}
                    </For>
                </div>
            </section>

            <section>
                <h4>Relations</h4>

                <AutoComplete<Relation>
                    value=""
                    items={props.relations}
                    getLabel={(relation) => {
                        const source =
                            props.entities.find((entity) => entity.id === relation.sourceId)
                                ?.label ?? relation.sourceId;

                        const target =
                            props.entities.find((entity) => entity.id === relation.targetId)
                                ?.label ?? relation.targetId;

                        return `${ source } → ${ relation.type } → ${ target }`;
                    }}
                    onInput={() => { }}
                    onSelect={addRelation}
                    placeholder="Add relation"
                />

                <div class="row wrap tiny-space">
                    <For each={selectedRelations()}>
                        {(relation) => (
                            <span class="small chip left-padding">
                                {relation.type}

                                <button
                                    type="button"
                                    class="transparent small circle no-padding"
                                    onClick={() => removeRelation(relation.id)}
                                >
                                    <i class="small">close</i>
                                </button>
                            </span>
                        )}
                    </For>
                </div>
            </section>

            <div class="field textarea border">
                <textarea
                    value={quote()}
                    disabled={saving()}
                    onInput={(event) => setQuote(event.currentTarget.value)}
                />
                <label>Quote</label>
            </div>

            <div class="field textarea border">
                <textarea
                    required
                    value={observation()}
                    disabled={saving()}
                    onInput={(event) => setObservation(event.currentTarget.value)}
                />
                <label>Observation</label>
            </div>

            <div class="field border">
                <select
                    value={status()}
                    disabled={saving()}
                    onChange={(event) =>
                        setStatus(event.currentTarget.value as EvidenceStatus)
                    }
                >
                    <For each={evidenceStatuses}>
                        {(value) => <option value={value}>{value}</option>}
                    </For>
                </select>

                <output>Status</output>
            </div>

            <div class="field textarea border">
                <textarea
                    value={notes()}
                    disabled={saving()}
                    onInput={(event) => setNotes(event.currentTarget.value)}
                />
                <label>Notes</label>
            </div>

            <nav class="footer">
                <button
                    type="submit"
                    disabled={saving() || !sourceId() || !observation().trim()}
                >
                    {saving() ? "Saving…" : editing() ? "Save" : "Add"}
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
        </form>
    );
}
