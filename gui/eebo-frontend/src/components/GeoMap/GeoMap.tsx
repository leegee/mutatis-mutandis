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
    label: string;  // canonical display name
    rawPlace?: string; // original EEBO imprint string
    count: number; // number of documents
    resolution?: "city" | "country" | "unknown";
};

export default function GeoMap(props: ClusterGeoMapProps) {
    let mapContainer: HTMLDivElement | undefined;
    let map: maplibregl.Map | undefined;
    const [mapReady, setMapReady] = createSignal(false);
    const markers: maplibregl.Marker[] = [];

    const totalDocuments = () => props.points.reduce(
        (sum, p) => sum + p.count,
        0
    );

    const resolvedPlaces = () => props.points.length;

    function clearMarkers() {
        for (const m of markers) m.remove();
        markers.length = 0;
    }

    function renderPoints(points: EventPoint[]) {
        if (!map) return;

        clearMarkers();

        for (const p of points) {
            const size = Math.min(6 + Math.sqrt(p.count) * 2.5, 30);
            const label = p.label || p.rawPlace || "Unknown";
            const el = document.createElement("div");
            el.className = "geo-event";
            el.style.width = `${ size }px`;
            el.style.height = `${ size }px`;
            el.innerText = String(p.count);
            el.innerHTML = `
                <div class="geo-circle" style=" width:${ size }px; height:${ size }px; ">
                ${ p.count }
                </div>
                <div class="geo-label">
                ${ label }
                </div>
            `;
            const marker = new maplibregl.Marker({ element: el })
                .setLngLat([p.lng, p.lat])
                .setPopup(
                    new maplibregl.Popup({ offset: 12 })
                        .setHTML(`
                            <strong>${ label }</strong><br/>
                            Documents: ${ p.count }
                            ${ p.rawPlace ? `<br/>EEBO: ${ p.rawPlace }` : "" }
                            `)
                )
                .addTo(map);

            markers.push(marker);
        }
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

    return (
        <div class="map-wrapper">
            <div ref={el => (mapContainer = el)}
                class="map-container" />

            <div class="map-info">
                <div>
                    Places: {resolvedPlaces()}
                </div>
                <div>
                    Documents: {totalDocuments()}
                </div>
            </div>
        </div>
    );
}
