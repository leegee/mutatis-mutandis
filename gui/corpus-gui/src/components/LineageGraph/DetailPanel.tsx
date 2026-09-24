import { For, Show } from "solid-js";
import type { LineageEvent, LineageNode } from "~/types/lineage";
import styles from "./LineageGraph.module.css";

type DetailPanelProps = {
    node: LineageNode;
    concept?: string;
    onClose: () => void;
};

type EventGroup = {
    doc_id: LineageEvent["doc_id"];
    events: LineageEvent[];
};

function groupEventsByDocument(events: LineageEvent[]): EventGroup[] {
    const groups = new Map<LineageEvent["doc_id"], LineageEvent[]>();

    for (const event of events) {
        const existing = groups.get(event.doc_id);

        if (existing) {
            existing.push(event);
        } else {
            groups.set(event.doc_id, [event]);
        }
    }

    return Array.from(groups, ([doc_id, groupedEvents]) => ({
        doc_id,
        events: groupedEvents,
    }));
}

function ContextProfile(props: {
    profile: LineageNode["context_profile"];
}) {
    return (
        <Show when={props.profile.length}>
            <section class={styles.contextProfile}>
                <h6>Characteristic vocabulary</h6>

                <ul class="list no-space border small-text">
                    <For each={props.profile}>
                        {(entry) => (
                            <li class="padding">
                                <span class="code large-text">
                                    {entry.token}
                                </span>

                                <span class={styles.neighbourMeta}>
                                    {" · "}
                                    {entry.count}{" "}
                                    {entry.count === 1
                                        ? "occurrence"
                                        : "occurrences"}
                                    {" · "}
                                    distinctiveness{" "}
                                    {entry.score.toFixed(2)}
                                </span>
                            </li>
                        )}
                    </For>
                </ul>
            </section>
        </Show>
    );
}


function EventSampleItem(props: { event: LineageEvent }) {
    const event = props.event;

    const documentUrl = () =>
        `/corpus/document/${ encodeURIComponent(event.doc_id) }?token_idx=${ event.token_idx }`;

    return (
        <div class={styles.eventSample}>
            <a
                href={documentUrl()}
                target="_blank"
                rel="noreferrer"
            >
                <strong>{event.token}</strong>
            </a>

            <span class={styles.neighbourMeta}>
                {" · "}
                {event.doc_id}
                {" · "}
                {event.token_idx}
            </span>
        </div>
    );
}


function DocumentGroup(props: { group: EventGroup }) {
    const group = props.group;

    return (
        <div>
            <button
                type="button"
                class="row transparent small-text"
                aria-expanded="true"
            >
                <i>expand_less</i>

                <strong>{group.doc_id}</strong>

                <span>
                    {" · "}
                    {group.events.length}{" "}
                    {group.events.length === 1 ? "event" : "events"}
                </span>
            </button>

            <div class={styles.documentEvents}>
                <For each={group.events}>
                    {(event) => <EventSampleItem event={event} />}
                </For>
            </div>
        </div>
    );
}

export default function DetailPanel(props: DetailPanelProps) {
    const eventGroups = () =>
        groupEventsByDocument(props.node.event_sample ?? []);

    return (
        <aside
            class={`${ styles.detailPanel } no-margin left-padding right-padding surface-container-high`}
        >
            <header
                class={`${ styles.detailPanelHeader } middle-align transparent`}
            >
                <h6 class="max medium-text">
                    <strong>{props.concept}</strong>
                    {" · "}
                    {props.node.year}

                    <span class="max small-text">
                        {" · cluster "}
                        {props.node.cluster}
                    </span>
                </h6>

                <button
                    type="button"
                    class={styles.detailPanelClose}
                    onClick={props.onClose}
                    aria-label="Close cluster details"
                >
                    <i>close</i>
                </button>
            </header>

            <div class={styles.detailPanelMeta}>
                <span>mass {props.node.size}</span>

                <Show when={props.node.persistence_score !== undefined}>
                    <span>
                        persistence{" "}
                        {props.node.persistence_score.toFixed(2)}
                    </span>
                </Show>

                <Show when={props.node.lineage_stable === false}>
                    <span class={styles.driftedTag}>
                        drifted lineage
                    </span>
                </Show>

                <Show when={props.node.merged_from.length}>
                    <span>
                        merged from lineage
                        {props.node.merged_from.length > 1 ? "s" : ""}{" "}
                        {props.node.merged_from.join(", ")}
                    </span>
                </Show>
            </div>

            <ContextProfile profile={props.node.context_profile} />

            <section>
                <h6>Sampled events</h6>

                <Show
                    when={eventGroups().length}
                    fallback={
                        <p class={styles.detailPanelEmpty}>
                            No sampled events.
                        </p>
                    }
                >
                    <div class="no-space">
                        <For each={eventGroups()}>
                            {(group) => <DocumentGroup group={group} />}
                        </For>
                    </div>
                </Show>
            </section>
        </aside>
    );
}
