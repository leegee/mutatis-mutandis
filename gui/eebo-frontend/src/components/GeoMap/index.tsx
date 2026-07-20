import { createMemo, createResource } from "solid-js";
import GeoMap, { type EventPoint } from "./GeoMap";
import { controls } from "../../state/controls.store";
import { fetchEventsGeo, type EventGeoRow } from "../../services/db";
import ControlsHeader from "../ControlsHeader";


export function aggregatePlaces(rows: EventGeoRow[]): EventPoint[] {
    const map = new Map<
        string,
        EventPoint
    >();

    for (const r of rows) {
        if (r.lat == null || r.lng == null) continue;

        const label = Array.isArray(r.normalized_place)
            ? r.normalized_place[0]
            : r.normalized_place;

        if (!label) continue;

        // stable spatial key
        const key = `${ r.lat.toFixed(6) },${ r.lng.toFixed(6) }`;

        const existing = map.get(key);

        if (!existing) {
            map.set(key, {
                lat: r.lat,
                lng: r.lng,
                label,
                count: 1,
            });
        } else {
            existing.count += 1;
        }
    }

    return [...map.values()];
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
        const rows = events() ?? [];

        console.log("[geo] rows", rows.length);

        const points = aggregatePlaces(rows);

        console.log("[geo] points", points.length);

        return points;
    });

    return (
        <>
            <ControlsHeader />
            <GeoMap points={mapPoints()} />
        </>
    );
}
