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
    count: number;
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
            const size = Math.min(6 + Math.sqrt(p.count) * 2.5, 30);

            const el = document.createElement("div");
            el.className = "geo-event";
            el.style.width = `${ size }px`;
            el.style.height = `${ size }px`;
            el.innerText = String(p.count);
            el.innerHTML = `
  <div style="
    width:${ size }px;
    height:${ size }px;
    border-radius:50%;
    background:#3b82f6;
    opacity:0.75;
    display:flex;
    align-items:center;
    justify-content:center;
    color:white;
    font-size:10px;
    font-weight:600;
  ">${ p.count }</div>
  <div style="text-align:center; margin-top:2px; text-shadow:0 0 2px black; ">
  ${ JSON.parse(p.label) }
  </div>
`;
            const marker = new maplibregl.Marker({ element: el })
                .setLngLat([p.lng, p.lat])
                .setPopup(
                    new maplibregl.Popup({ offset: 12 }).setText(
                        `${ p.label || "-" } (${ p.count || "-" })`
                    )
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

    return <div ref={el => (mapContainer = el)} style="width: 100%; height: 100%;" />;
}
