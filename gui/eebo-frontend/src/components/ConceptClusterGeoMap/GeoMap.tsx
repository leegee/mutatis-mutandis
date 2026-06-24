import { onMount, onCleanup, createEffect, createSignal } from "solid-js";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

import './GeoMap.css';

interface ClusterGeoMapProps {
    points: EventPoint[];
}

export type EventPoint = {
    lat: number;
    lng: number;
    label: string;
};

export default function GeoMap(props: ClusterGeoMapProps) {
    let mapContainer: HTMLDivElement | undefined;
    let map: maplibregl.Map | undefined;
    const [mapReady, setMapReady] = createSignal(false);

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
            el.className = 'geo-event';

            const marker = new maplibregl.Marker({ element: el })
                .setLngLat([p.lng, p.lat])
                .setPopup(
                    new maplibregl.Popup({ offset: 12 }).setText(p.label ?? "")
                )
                .addTo(map);
            markers.push(marker);
        }
        console.log("[GeoMap] marker count", markers.length)
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
            setMapReady(true)
            renderPoints(props.points ?? []);
        });
    });

    createEffect(() => {
        if (!map || !mapReady()) return;
        renderPoints(props.points ?? []);
    });

    onCleanup(() => {
        clearMarkers();
        map?.remove();
    });

    return <div ref={el => (mapContainer = el)} style="width: 100%; height: 100%;" />;
}
