import { For, Show } from "solid-js";

import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

interface EntityInspectorProps {
    entity: Entity | undefined;
    entities: Entity[];
    relations: Relation[];
    onClose?: () => void;
}

export default function EntityInspector(
    props: EntityInspectorProps,
) {
    function entityLabel(id: string) {
        return (
            props.entities.find((entity) => entity.id === id)?.label ??
            id
        );
    }

    const outgoing = () =>
        props.relations.filter(
            (relation) =>
                relation.sourceId === props.entity?.id,
        );

    const incoming = () =>
        props.relations.filter(
            (relation) =>
                relation.targetId === props.entity?.id,
        );

    return (
        <aside class="padding">
            <Show
                when={props.entity}
                fallback={
                    <div>
                        <h3>No entity selected</h3>
                        <p>
                            Select an entity in the graph to inspect it.
                        </p>
                    </div>
                }
            >
                {(entity) => (
                    <>
                        <header>
                            <nav>
                                <div>
                                    <h3>{entity().label}</h3>
                                    <small>{entity().type}</small>
                                </div>

                                <button
                                    class="circle transparent"
                                    type="button"
                                    title="Close"
                                    onClick={props.onClose}
                                >
                                    ×
                                </button>
                            </nav>
                        </header>

                        <Show when={entity().description}>
                            <p>{entity().description}</p>
                        </Show>

                        <Show when={entity().aliases?.length}>
                            <section>
                                <h5>Aliases</h5>

                                <For each={entity().aliases}>
                                    {(alias) => (
                                        <span class="chip">
                                            {alias}
                                        </span>
                                    )}
                                </For>
                            </section>
                        </Show>

                        <Show when={entity().tags?.length}>
                            <section>
                                <h5>Tags</h5>

                                <div class="row wrap">
                                    <For each={entity().tags}>
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

                            <Show
                                when={
                                    outgoing().length > 0 ||
                                    incoming().length > 0
                                }
                                fallback={
                                    <p>No relationships yet.</p>
                                }
                            >
                                <Show when={outgoing().length > 0}>
                                    <h6>Outgoing</h6>

                                    <ul>
                                        <For each={outgoing()}>
                                            {(relation) => (
                                                <li>
                                                    <strong>
                                                        {relation.type}
                                                    </strong>
                                                    {" → "}
                                                    {entityLabel(
                                                        relation.targetId,
                                                    )}
                                                </li>
                                            )}
                                        </For>
                                    </ul>
                                </Show>

                                <Show when={incoming().length > 0}>
                                    <h6>Incoming</h6>

                                    <ul>
                                        <For each={incoming()}>
                                            {(relation) => (
                                                <li>
                                                    {entityLabel(
                                                        relation.sourceId,
                                                    )}
                                                    {" → "}
                                                    <strong>
                                                        {relation.type}
                                                    </strong>
                                                </li>
                                            )}
                                        </For>
                                    </ul>
                                </Show>
                            </Show>
                        </section>
                    </>
                )}
            </Show>
        </aside>
    );
}
