import {
    createEffect,
    createSignal,
    onCleanup,
    onMount,
} from "solid-js";

import cytoscape, {
    type Core,
    type ElementDefinition,
} from "cytoscape";

import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

interface GraphViewProps {
    entities: Entity[];
    relations: Relation[];
    onSelectEntity?: (entity: Entity) => void;
    onSelectRelation?: (relation: Relation) => void;
}

export default function GraphView(
    props: GraphViewProps,
) {
    let container!: HTMLDivElement;

    const [cy, setCy] = createSignal<Core>();

    function buildElements(): ElementDefinition[] {
        const nodes: ElementDefinition[] =
            props.entities.map((entity) => ({
                data: {
                    id: entity.id,
                    label: entity.label,
                    type: entity.type,
                },
            }));

        const edges: ElementDefinition[] =
            props.relations
                .filter(
                    (relation) =>
                        props.entities.some(
                            (entity) => entity.id === relation.sourceId,
                        ) &&
                        props.entities.some(
                            (entity) => entity.id === relation.targetId,
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

                        "background-color": "#455a64",
                        color: "#ffffff",

                        "border-width": 2,
                        "border-color": "#90a4ae",

                        "font-size": "12px",
                        "font-weight": 500,

                        width: "44px",
                        height: "44px",
                    },
                },

                {
                    selector: "node:selected",
                    style: {
                        "background-color": "#78909c",
                        "border-width": 3,
                        "border-color": "#ffffff",
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
                    },
                },
            ],

            layout: {
                name: "cose",
                animate: false,
            },
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
            const id = event.target.id();

            const entity = props.entities.find(
                (item) => item.id === id,
            );

            if (entity) {
                props.onSelectEntity?.(entity);
            }
        });

        setCy(instance);
    });

    createEffect(() => {
        const instance = cy();

        if (!instance) {
            return;
        }

        instance.elements().remove();
        instance.add(buildElements());

        instance
            .layout({
                name: "cose",
                animate: false,
            })
            .run();
    });

    onCleanup(() => {
        cy()?.destroy();
    });

    return (
        <div
            ref={container}
            style={{
                width: "100%",
                height: "70vh",
                "min-height": "500px",
            }}
        />
    );
}
