import { onMount, onCleanup, createEffect } from "solid-js";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

import type { ConceptDataset } from "../ScatterPlot/types";

interface ClusterGeoMapProps {
    clusterDatasets: ConceptDataset[] | undefined;
}

export type EventPoint = {
    lat: number;
    lng: number;
    label: string;
};

export default function GeoMap(props: ClusterGeoMapProps) {
    let mapContainer: HTMLDivElement | undefined;
    let map: maplibregl.Map | undefined;

    const markers: maplibregl.Marker[] = [];

    function clearMarkers() {
        for (const m of markers) m.remove();
        markers.length = 0;
    }

    function renderPoints(points: EventPoint[]) {
        if (!map) return;

        clearMarkers();

        for (const p of points) {
            if (p.lat == null || p.lng == null) continue;

            const el = document.createElement("div");
            el.style.width = "8px";
            el.style.height = "8px";
            el.style.borderRadius = "50%";
            el.style.background = "#3b82f6";
            el.style.opacity = "0.8";

            const marker = new maplibregl.Marker({ element: el })
                .setLngLat([p.lng, p.lat])
                .setPopup(
                    new maplibregl.Popup({ offset: 12 }).setText(p.label ?? "")
                )
                .addTo(map);

            markers.push(marker);
        }
    }

    function extractPoints(): EventPoint[] {
        const datasets = props.clusterDatasets ?? [];

        return datasets.flatMap(ds =>
            (ds.points ?? [])
                .map((p: any) => ({
                    lat: p.lat,
                    lng: p.lng,
                    label: p.placename ?? p.label ?? "unknown"
                }))
                .filter((p: EventPoint) => p.lat != null && p.lng != null)
        );
    }

    onMount(() => {
        if (!mapContainer) return;

        map = new maplibregl.Map({
            container: mapContainer,
            style: "https://tiles.stadiamaps.com/styles/alidade_smooth_dark.json",
            center: [-2.8, 54.4],
            zoom: 5.6,
            minZoom: 3,
            maxZoom: 14
        });

        map.on("load", () => {
            renderPoints(extractPoints());
        });
    });

    createEffect(() => {
        renderPoints(extractPoints());
    });

    onCleanup(() => {
        clearMarkers();
        map?.remove();
    });

    return <div ref={el => (mapContainer = el)} style="width: 100%; height: 100%;" />;
}
