import {
    createEffect,
    createSignal,
    Match,
    onCleanup,
    onMount,
    Show,
    Switch,
} from "solid-js";

import cytoscape, {
    type Core,
    type ElementDefinition,
} from "cytoscape";

import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

type ContextMenu =
    | {
        kind: "canvas";
        x: number;
        y: number;
    }
    | {
        kind: "node";
        x: number;
        y: number;
        nodeId: string;
    }
    | {
        kind: "edge";
        x: number;
        y: number;
        relationId: string;
    };

const [contextMenu, setContextMenu] = createSignal<ContextMenu>();

const [linkingFrom, setLinkingFrom] = createSignal<string>();

interface GraphViewProps {
    entities: Entity[];
    relations: Relation[];

    onSelectEntity?: (entity: Entity) => void;
    onSelectRelation?: (relation: Relation) => void;

    onAddEntity?: (position: {
        x: number;
        y: number;
    }) => void;

    onEditEntity?: (entity: Entity) => void;
    onDeleteEntity?: (entity: Entity) => void;

    onAddRelation?: (
        sourceId: string,
        targetId: string,
    ) => void;

    onEditRelation?: (relation: Relation) => void;
    onDeleteRelation?: (relation: Relation) => void;
}

const LAYOUT_PARAMS = {
    name: "cose",
    animate: false,
    nodeRepulsion: 8000,
    idealEdgeLength: 120,
    edgeElasticity: 100,
    nestingFactor: 1.2,
    gravity: 0.25,
};


function nodeSize(incoming: number): number {
    return 46 + Math.sqrt(incoming) * 12;
}

export default function GraphView(props: GraphViewProps,) {
    let container!: HTMLDivElement;

    const [cy, setCy] = createSignal<Core>();

    function buildElements(): ElementDefinition[] {
        const incomingCounts = new Map<string, number>();

        for (const relation of props.relations) {
            incomingCounts.set(
                relation.targetId,
                (incomingCounts.get(relation.targetId) ?? 0) + 1,
            );
        }

        const nodes: ElementDefinition[] =
            props.entities.map((entity) => {
                const incoming =
                    incomingCounts.get(entity.id) ?? 0;

                return {
                    data: {
                        id: entity.id,
                        label: entity.label,
                        type: entity.type,
                        incoming,
                        size: nodeSize(incoming),
                    },
                };
            });

        const edges: ElementDefinition[] =
            props.relations
                .filter(
                    (relation) =>
                        props.entities.some(
                            (entity) =>
                                entity.id === relation.sourceId,
                        ) &&
                        props.entities.some(
                            (entity) =>
                                entity.id === relation.targetId,
                        ),
                )
                .map((relation) => ({
                    data: {
                        id: relation.id,
                        source: relation.sourceId,
                        target: relation.targetId,
                        label: relation.type,
                    },
                }));

        return [...nodes, ...edges];
    }

    onMount(() => {
        const instance = cytoscape({
            container,

            elements: buildElements(),

            style: [
                {
                    selector: "node",
                    style: {
                        label: "data(label)",
                        "text-valign": "center",
                        "text-halign": "center",

                        "background-color": "#37474f",
                        color: "#ffffff",
                        "border-width": 2,
                        "border-color": "#78909c",
                        "font-size": "12px",
                        "font-weight": 500,
                        "text-wrap": "wrap",
                        "text-max-width": "80px",

                        width: "data(size)",
                        height: "data(size)",
                    },
                },

                // Concepts
                {
                    selector: 'node[type = "concept"]',
                    style: {
                        "background-color": "#455a64",
                        "border-color": "#90a4ae",
                    },
                },

                // Lexical forms
                {
                    selector: 'node[type = "lexeme"]',
                    style: {
                        "background-color": "#4e5d6c",
                        "border-color": "#9fa8b2",
                    },
                },

                // Motifs
                {
                    selector: 'node[type = "motif"]',
                    style: {
                        "background-color": "#51445f",
                        "border-color": "#b39ddb",
                    },
                },

                // Animals
                {
                    selector: 'node[type = "animal"]',
                    style: {
                        "background-color": "#455a50",
                        "border-color": "#81a995",
                    },
                },

                // People
                {
                    selector: 'node[type = "person"]',
                    style: {
                        "background-color": "#5a4b42",
                        "border-color": "#bcaaa4",
                    },
                },

                // Sources
                {
                    selector: 'node[type = "source"]',
                    style: {
                        "background-color": "#4a5060",
                        "border-color": "#9fa8da",
                    },
                },

                {
                    selector: "node:selected",
                    style: {
                        "background-color": "#10063f",
                        "border-width": 3,
                        "border-color": "#ffffff",
                        "color": "#ffffff",
                        "overlay-color": "#26084d",
                        "overlay-opacity": 0.08,
                    },
                },

                {
                    selector: "edge",
                    style: {
                        width: 2,

                        "line-color": "#90a4ae",

                        "target-arrow-color": "#90a4ae",
                        "target-arrow-shape": "triangle",

                        "curve-style": "bezier",

                        label: "data(label)",

                        color: "#eeeeee",
                        "text-background-color": "#263238",
                        "text-background-opacity": 1,
                        "text-background-padding": "3px",

                        "font-size": "10px",
                        "font-weight": 500,
                    },
                },

                {
                    selector: "edge:selected",
                    style: {
                        width: 3,

                        "line-color": "#ffffff",
                        "target-arrow-color": "#ffffff",
                        color: "#ffffff",
                        "text-background-color": "#37474f",
                    },
                },

                {
                    selector: "node.link-source",
                    style: {
                        "border-width": 4,
                        "border-color": "#ffffff",
                        "overlay-color": "#ffffff",
                        "overlay-opacity": 0.15,
                    },
                },

                {
                    selector: "node.link-target",
                    style: {
                        "border-width": 3,
                        "border-color": "#ffffff",
                    },
                },
            ],

            layout: LAYOUT_PARAMS,
        });


        instance.on("cxttap", (event) => {
            event.originalEvent.preventDefault();

            const rect = container.getBoundingClientRect();
            const x = event.originalEvent.clientX - rect.left;
            const y = event.originalEvent.clientY - rect.top;

            if (event.target === instance) {
                setContextMenu({
                    kind: "canvas",
                    x,
                    y,
                });

                return;
            }

            if (event.target.isNode()) {
                setContextMenu({
                    kind: "node",
                    x,
                    y,
                    nodeId: event.target.id(),
                });

                return;
            }

            if (event.target.isEdge()) {
                setContextMenu({
                    kind: "edge",
                    x,
                    y,
                    relationId: event.target.id(),
                });
            }
        });


        instance.on("tap", "edge", (event) => {
            const relationId = event.target.id();

            const relation = props.relations.find(
                (relation) =>
                    relation.id === relationId,
            );

            if (relation) {
                props.onSelectRelation?.(relation);
            }
        });

        instance.on("tap", "node", (event) => {
            const targetId = event.target.id();
            const sourceId = linkingFrom();

            if (sourceId) {
                if (sourceId !== targetId) {
                    setLinkingFrom(undefined);

                    instance
                        .nodes()
                        .removeClass("link-target");

                    instance
                        .getElementById(sourceId)
                        .removeClass("link-source");

                    const target = props.entities.find(
                        (entity) =>
                            entity.id === targetId,
                    );

                    if (target) {
                        props.onAddRelation?.(
                            sourceId,
                            targetId,
                        );
                    }
                }

                return;
            }

            const entity = props.entities.find(
                (item) => item.id === targetId,
            );

            if (entity) {
                props.onSelectEntity?.(entity);
            }
        });

        instance.on("mouseover", "node", (event) => {
            const node = event.target;
            node.style({
                "font-size": 32,
                "font-weight": 400,
                "z-index": 9999,
            });
        });

        instance.on("mouseout", "node", (event) => {
            const node = event.target;
            node.style({
                "font-size": 12,
                "font-weight": 500,
                "z-index": 0,
            });
        });

        instance.on("mouseover", "edge", (event) => {
            const edge = event.target;
            edge.style({
                "font-size": 32,
                "font-weight": 600,
                "z-index": 999999,
            });

        });

        instance.on("mouseout", "edge", (event) => {
            const edge = event.target;
            edge.style({
                "font-size": 10,
                "font-weight": 500,
                "z-index": 0,
            });

        });

        setCy(instance);

        const handleKeyDown = (event: KeyboardEvent,) => {
            if (event.key !== "Escape") {
                return;
            }

            setContextMenu(undefined);

            const sourceId = linkingFrom();

            if (sourceId) {
                setLinkingFrom(undefined);
                instance.getElementById(sourceId).removeClass("link-source");
                instance.nodes().removeClass("link-target");
            }
        };

        window.addEventListener("keydown", handleKeyDown,);
        onCleanup(() => { window.removeEventListener("keydown", handleKeyDown,); });
    });

    createEffect(() => {
        const instance = cy();

        if (!instance) {
            return;
        }

        instance.elements().remove();
        instance.add(buildElements());

        instance
            .layout(LAYOUT_PARAMS)
            .run();
    });

    onCleanup(() => {
        cy()?.destroy();
    });

    return (
        <div
            ref={container}
            style={{
                position: "relative",
                width: "100%",
                height: "90vh",
                "min-height": "500px",
            }}
            onClick={() => setContextMenu(undefined)}
        >
            <Show when={contextMenu()}>
                {(menu) => (
                    <div
                        class="graph-context-menu"
                        style={{
                            position: "absolute",
                            left: `${ menu().x }px`,
                            top: `${ menu().y }px`,
                            "z-index": 100000,
                        }}
                        onClick={(event) =>
                            event.stopPropagation()
                        }
                    >
                        <menu class="active group no-wrap small-space top">

                            <Switch>
                                <Match when={menu().kind === "canvas"} >
                                    <button type="button" class="fill"
                                        onClick={() => {
                                            const item =
                                                menu();

                                            if (item.kind !== "canvas") {
                                                return;
                                            }

                                            props.onAddEntity?.({
                                                x: item.x,
                                                y: item.y,
                                            });

                                            setContextMenu(undefined,);
                                        }}
                                    >
                                        Add node
                                    </button>
                                </Match>

                                <Match when={menu().kind === "node"} >
                                    <button type="button" class="fill"
                                        onClick={() => {
                                            const item = menu();

                                            if (item.kind !== "node") {
                                                return;
                                            }

                                            const entity = props.entities.find(
                                                (entity) => entity.id === item.nodeId
                                            );

                                            if (entity) {
                                                props.onEditEntity?.(entity,);
                                            }

                                            setContextMenu(undefined,);
                                        }}
                                    >
                                        Edit node
                                    </button>

                                    <button type="button" class="fill"
                                        onClick={() => {
                                            const item = menu();
                                            if (item.kind !== "node") {
                                                return;
                                            }
                                            setLinkingFrom(item.nodeId);
                                            const instance = cy()!;

                                            instance.getElementById(item.nodeId).addClass("link-source");
                                            instance.nodes()
                                                .not(
                                                    `#${ CSS.escape(
                                                        item.nodeId,
                                                    ) }`,
                                                )
                                                .addClass("link-target",);

                                            setContextMenu(undefined);
                                        }}
                                    >
                                        Add relation →
                                    </button>

                                    <button type="button" class="error-container on-error"
                                        onClick={() => {
                                            const item =
                                                menu();

                                            if (item.kind !== "node") {
                                                return;
                                            }

                                            const entity = props.entities.find(
                                                (entity) => entity.id === item.nodeId
                                            );

                                            if (entity) {
                                                props.onDeleteEntity?.(entity);
                                            }

                                            setContextMenu(undefined);
                                        }}
                                    >
                                        Delete node
                                    </button>
                                </Match>

                                <Match when={menu().kind === "edge"} >
                                    <button type="button"
                                        onClick={() => {
                                            const item = menu();

                                            if (item.kind !== "edge") {
                                                return;
                                            }

                                            const relation = props.relations.find(
                                                (relation) => relation.id === item.relationId
                                            );

                                            if (relation) {
                                                props.onEditRelation?.(relation);
                                            }

                                            setContextMenu(undefined);
                                        }}
                                    >
                                        Edit relation
                                    </button>

                                    <button
                                        type="button"
                                        class="danger"
                                        onClick={() => {
                                            const item = menu();

                                            if (item.kind !== "edge") {
                                                return;
                                            }

                                            const relation = props.relations.find(
                                                (relation) => relation.id === item.relationId
                                            );

                                            if (relation) {
                                                props.onDeleteRelation?.(relation);
                                            }

                                            setContextMenu(undefined);
                                        }}
                                    >
                                        Delete relation
                                    </button>
                                </Match>
                            </Switch>

                        </menu>
                    </div>
                )}
            </Show>

            <Show when={linkingFrom()}>
                <div
                    class="graph-linking-indicator fill padding round"
                    style={{
                        position: "absolute",
                        bottom: "2em",
                        left: "50%",
                        transform: "translateX(-50%)",
                        "z-index": 100000,
                    }}
                >
                    Click a node to create the relation. Press Escape to cancel.
                </div>
            </Show>
        </div>
    );
}
