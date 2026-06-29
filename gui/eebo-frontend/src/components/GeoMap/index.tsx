import { createMemo, createResource } from "solid-js";
import GeoMap, { type EventPoint } from "./GeoMap";
import { controls } from "../../state/controls.store";
import { fetchEvents } from "../../services/db";
import ControlsHeader from "../ControlsHeader";

type RawRow = {
    event_id: bigint;
    doc_id: string;
    pub_year: number;
    token: string;
    geom: string;
    lat: number;
    lng: number;
    label: string;
};

export function aggregatePlaces(rows: RawRow[]) {
    const map = new Map<
        string,
        { lat: number; lng: number; label: String; count: number }
    >();

    for (const r of rows) {
        if (r.lat == null || r.lng == null) continue;

        // stable spatial key
        const key = `${ r.lat.toFixed(6) },${ r.lng.toFixed(6) }`;

        const existing = map.get(key);

        if (!existing) {
            map.set(key, {
                lat: r.lat,
                lng: r.lng,
                label: r.label,
                count: 1,
            });
        } else {
            existing.count += 1;
        }
    }

    return [...map.entries()].map(([_, v]) => v);
}

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
        const eventPoints = eventRows.map((eventPoint) => ({
            lat: eventPoint.lat,
            lng: eventPoint.lng,
            label: eventPoint.token ?? controls.conceptSelection[0],
            // type: "event"
        }));

        // console.debug("[geo] event", eventRows);
        // console.debug("[geo] event points", eventPoints);

        return aggregatePlaces([
            // ...clusterPoints,
            ...eventPoints as RawRow[]
        ]) as EventPoint[];
    });

    return (
        <>
            <ControlsHeader />
            <GeoMap points={mapPoints()} />
        </>
    );
}