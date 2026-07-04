import { createMemo, createResource } from "solid-js";
import GeoMap, { type EventPoint } from "./GeoMap";
import { controls } from "../../state/controls.store";
import { fetchEventsGeo } from "../../services/db";
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

    const [events] = createResource(
        sharedKey,
        fetchEventsGeo
    );

    const mapPoints = createMemo(() => {
        console.log("[geo] making map points");
        const eventRows = events() ?? [];
        const eventPoints = eventRows.map((eventPoint) => ({
            lat: eventPoint.lat,
            lng: eventPoint.lng,
            label: eventPoint.normalized_places,
        }));

        return aggregatePlaces([
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