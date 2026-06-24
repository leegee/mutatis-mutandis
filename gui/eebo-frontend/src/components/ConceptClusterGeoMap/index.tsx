import { createMemo, createResource } from "solid-js";
import GeoMap, { type EventPoint } from "./GeoMap";
import { controls } from "../../state/controls.store";
import { loadDatasets } from "../ScatterPlot/loadScatterDatasets";
import { buildEventQuery, fetchEvents } from "../../services/db";

export default function ConceptClusterGeoMap() {

    const sharedKey = () => ({
        concepts: controls.conceptSelection,
        fromYear: controls.fromYear,
        toYear: controls.toYear,
        yearMode: controls.yearMode,
    });

    // const [conceptDatasets] = createResource(
    //     () => ({ ...sharedKey(), dataType: "concept" }),
    //     loadDatasets
    // );

    const [events] = createResource(
        sharedKey,
        fetchEvents
    );

    const mapPoints = createMemo(() => {
        console.log("[geo] making map points");
        // const clusters = conceptDatasets() ?? [];
        // const clusterPoints = clusters.flatMap(ds =>
        //     (ds.points ?? []).map(p => ({
        //         lat: p.lat,
        //         lng: p.lng,
        //         label: p.placename ?? p.label ?? "cluster",
        //         type: "cluster"
        //     }))
        // );

        // console.debug("[geo] cluster points", clusterPoints);

        const eventRows = events() ?? [];
        const eventPoints = eventRows.map((e) => ({
            lat: e!.lat,
            lng: e!.lng,
            label: e!.token ?? "event",
            // type: "event"
        }));

        // console.debug("[geo] event", eventRows);
        // console.debug("[geo] event points", eventPoints);

        return [
            // ...clusterPoints,
            ...eventPoints
        ] as EventPoint[];
    });

    return <GeoMap points={mapPoints()} />;
}