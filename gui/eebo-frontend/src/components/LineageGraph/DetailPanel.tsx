import { createSignal, For, Show } from "solid-js";
import { showDocument } from "../../services/documentApi";
import TextWindow from "../TextWindow";
import styles from "./LineageGraph.module.css";
import type {
    EventSample,
    LineageNode,
    Neighbour,
} from "./types";


type DetailPanelProps = {
    node: LineageNode;
    concept?: string;
    onClose: () => void;
};

type EventGroup = {
    doc_id: EventSample["doc_id"];
    events: EventSample[];
};

function groupEventsByDocument(events: EventSample[]): EventGroup[] {
    const groups = new Map<EventSample["doc_id"], EventSample[]>();

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

function NeighbourItem(props: { neighbour: Neighbour }) {
    const nb = props.neighbour;

    return (
        <div class="padding">
            <h6>
                <span class="code large-text">{nb.token}</span>
                {" "}&mdash;{" "}
                <span class={styles.neighbourMeta}>
                    {"×"}
                    {nb.count}
                    {" · "}
                    {nb.max_score.toFixed(2)}
                </span>
            </h6>

            <ul class="list no-space border small-text">
                <For each={nb.examples}>
                    {(example) => (
                        <li>
                            {example.doc_id}
                            {" @ "}
                            {example.token_idx}
                            {" · "}
                            {example.score.toFixed(2)}
                        </li>
                    )}
                </For>
            </ul>
        </div>
    );
}

function EventSampleItem(props: { event: EventSample }) {
    const ev = props.event;
    const [showNeighbours, setShowNeighbours] = createSignal(false);

    return (
        <div>
            <div class="row">
                <button
                    type="button"
                    class="no-round transparent padding"
                    onDblClick={() => showDocument(ev.doc_id, ev.token_idx)}
                >
                    <TextWindow doc_id={String(ev.doc_id)} token_idx={ev.token_idx} />
                </button>
                <div class="tooltip top">Double click to view the text</div>
            </div>

            <Show when={ev.neighbours.length}>
                <div class="row">
                    <button
                        type="button"
                        class="chip transparent small-text responsive"
                        onClick={() => setShowNeighbours(!showNeighbours())}
                        aria-expanded={showNeighbours()}
                    >
                        <i>{showNeighbours() ? "expand_less" : "expand_more"}</i>
                        {showNeighbours() ? "Hide" : `Neighbours (${ ev.neighbours.length })`}
                    </button>
                </div>

                <Show when={showNeighbours()}>
                    <For each={ev.neighbours}>{(neighbour) => <NeighbourItem neighbour={neighbour} />}</For>
                </Show>
            </Show>
        </div>
    );
}



function DocumentGroup(props: { group: EventGroup }) {
    const group = props.group;
    const [expanded, setExpanded] = createSignal(true);
    return (
        <div>
            <button type="button" class={`row transparent small-text`}
                onClick={() => setExpanded(!expanded())} aria-expanded={expanded()} >
                <i> {expanded() ? "expand_less" : "expand_more"} </i>
                <strong>{group.doc_id}</strong>
                <span>
                    {" · "} {group.events.length}{" "} {group.events.length === 1 ? "event" : "events"}
                </span>
            </button>
            <Show when={expanded()}>
                <div class={styles.documentEvents}>
                    <For each={group.events}>
                        {(event) => (<EventSampleItem event={event} />)}
                    </For>
                </div>
            </Show>
        </div>);
}



export default function DetailPanel(props: DetailPanelProps) {
    const eventGroups = () => groupEventsByDocument(props.node.event_sample ?? []);

    return (
        <aside class={`${ styles.detailPanel } no-margin left-padding right-padding surface-container-high`}>
            <header class={`${ styles.detailPanelHeader } middle-align transparent`}>
                <h6 class="max medium-text">
                    <strong>{props.concept}</strong> · {props.node.year}
                    <span class="max small-text"> · cluster {props.node.cluster}</span>
                </h6>

                <button type="button" class={styles.detailPanelClose} onClick={props.onClose}>
                    <i>close</i>
                </button>
            </header>

            <div class={styles.detailPanelMeta}>
                <span>mass {props.node.size}</span>

                <Show when={props.node.persistence_score !== undefined}>
                    <span>persistence {props.node.persistence_score!.toFixed(2)}</span>
                </Show>

                <Show when={props.node.lineage_stable === false}>
                    <span class={styles.driftedTag}>drifted lineage</span>
                </Show>

                <Show when={props.node.merged_from?.length}>
                    <span>
                        merged from lineage
                        {props.node.merged_from!.length > 1 ? "s" : ""} {props.node.merged_from!.join(", ")}
                    </span>
                </Show>
            </div>

            <Show when={eventGroups().length} fallback={<p class={styles.detailPanelEmpty}>No sampled events.</p>}>
                <div class="no-space">
                    <For each={eventGroups()}>{(group) => <DocumentGroup group={group} />}</For>
                </div>
            </Show>
        </aside>
    );
}
