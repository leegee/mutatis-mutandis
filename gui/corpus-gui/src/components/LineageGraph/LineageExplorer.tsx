import { createEffect, createSignal, on, onCleanup, Show } from "solid-js";
import { getLineage } from "~/server/tier3/lineage";
import type { LineageData } from "~/types/lineage";
import { controls } from "../../state/controls.store";
import SingleConceptSelect from "../ControlsHeader/SingleConceptSelect";
import styles from "./LineageExplorer.module.css";
import LineageGraph from "./LineageGraph";
import type { ScrollState, ViewportRatio } from "./types";

export default function LineageExplorer() {
    const [data, setData] = createSignal<LineageData>();
    const [viewport, setViewport] = createSignal<ViewportRatio>({
        startRatio: 0,
        endRatio: 1,
    });

    // Not a signal on purpose -- this is an imperative handle to the
    // detail view's scrollable DOM node, used only inside
    // handleNavigate's event handler, not read reactively.
    let detailContainer: HTMLDivElement | undefined;

    createEffect(
        on(
            () => controls.conceptSelection[0],
            (concept) => {
                if (!concept) return;

                let cancelled = false;

                onCleanup(() => {
                    cancelled = true;
                });

                getLineage(concept)
                    .then((data) => {
                        if (cancelled) return;

                        setData(data);
                        setViewport({
                            startRatio: 0,
                            endRatio: 1,
                        });
                        detailContainer?.scrollTo({
                            left: 0,
                            top: 0,
                        });
                    })
                    .catch((err) => {
                        if (!cancelled) {
                            console.error(`[lineage] failed to load ${ concept }`, err);
                        }
                    });
            },
        ),
    );

    function handleViewportChange(state: ScrollState) {
        if (state.scrollWidth <= state.clientWidth) {
            setViewport({ startRatio: 0, endRatio: 1 });
            return;
        }

        setViewport({
            startRatio: state.scrollLeft / state.scrollWidth,
            endRatio: (state.scrollLeft + state.clientWidth) / state.scrollWidth,
        });
    }

    function handleNavigate(ratio: number) {
        if (!detailContainer) return;

        // Centre the detail view on the clicked point rather than
        // snapping its left edge there.
        const target = ratio * detailContainer.scrollWidth - detailContainer.clientWidth / 2;

        detailContainer.scrollTo({
            left: Math.max(0, Math.min(target, detailContainer.scrollWidth - detailContainer.clientWidth)),
            behavior: "smooth",
        });
    }

    return (
        <div class={styles.explorer}>
            <Show when={data()}>
                {(graph) => (
                    <>
                        <nav>
                            <SingleConceptSelect />
                            <LineageGraph data={graph()} variant="overview" viewport={viewport()} onNavigate={handleNavigate} />
                        </nav>
                        <LineageGraph
                            data={graph()}
                            variant="detail"
                            onContainerReady={(el) => (detailContainer = el)}
                            onViewportChange={handleViewportChange}
                        />
                    </>
                )}
            </Show>
        </div>
    );
}
