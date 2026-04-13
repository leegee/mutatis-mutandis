import { createEffect, onMount } from "solid-js";
import type { SliceView } from "../types";

type Neighbor = SliceView["neighbors"][number];

type Props = {
    slice: SliceView;
    width: number;
    height: number;
};

function gaussian(d2: number, sigma: number) {
    const s2 = sigma * sigma;
    return Math.exp(-d2 / (2 * s2));
}

function buildField(neighbors: Neighbor[]) {
    const points = new Array(neighbors.length);

    for (let i = 0; i < neighbors.length; i++) {
        const n = neighbors[i];
        const sim = n.similarity ?? 0;
        const count = n.count ?? 1;

        const mass = Math.log1p(count);

        points[i] = {
            x: sim,
            y: mass * sim,
            mass,
            sim
        };
    }

    return points;
}

function fieldAt(
    x: number,
    y: number,
    points: { x: number; y: number; mass: number; sim: number }[]
) {
    let sum = 0;

    for (let i = 0; i < points.length; i++) {
        const p = points[i];

        const dx = x - p.x;
        const dy = y - p.y;

        const d2 = dx * dx + dy * dy;

        const sigma = 0.08 + (1 - p.sim) * 0.2;

        sum += p.mass * gaussian(d2, sigma);
    }

    return sum;
}

function renderField(
    canvas: HTMLCanvasElement,
    neighbors: Neighbor[],
    width: number,
    height: number
) {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const image = ctx.createImageData(width, height);
    const data = image.data;

    const points = buildField(neighbors);

    const values = new Float32Array(width * height);

    let max = 0;

    // Field evaluation pass
    for (let y = 0; y < height; y++) {
        const ny = y / height;

        for (let x = 0; x < width; x++) {
            const nx = x / width;

            const v = fieldAt(nx, ny, points);

            const idx = y * width + x;
            values[idx] = v;

            if (v > max) max = v;
        }
    }

    const invMax = max > 0 ? 1 / max : 1;

    // Field render pass
    for (let i = 0; i < values.length; i++) {
        const v = values[i] * invMax;

        const hue = 120 + v * 160;
        const lightness = 30 + v * 40;

        // lightweight HSL approximation (no allocation)
        const c = (1 - Math.abs(2 * (lightness / 100) - 1));
        const hp = hue / 60;
        const x = c * (1 - Math.abs((hp % 2) - 1));

        let r = 0, g = 0, b = 0;

        if (hp < 1) [r, g, b] = [c, x, 0];
        else if (hp < 2) [r, g, b] = [x, c, 0];
        else if (hp < 3) [r, g, b] = [0, c, x];
        else if (hp < 4) [r, g, b] = [0, x, c];
        else if (hp < 5) [r, g, b] = [x, 0, c];
        else[r, g, b] = [c, 0, x];

        const o = i * 4;
        data[o] = r * 255;
        data[o + 1] = g * 255;
        data[o + 2] = b * 255;
        data[o + 3] = 255;
    }

    ctx.putImageData(image, 0, 0);
}

export default function SliceDensityField(props: Props) {
    let canvas: HTMLCanvasElement | undefined;

    const resize = () => {
        if (!canvas) return;
        canvas.width = props.width;
        canvas.height = props.height;
    };

    onMount(resize);

    createEffect(() => {
        if (!canvas) return;

        const neighbors = props.slice.neighbors ?? [];

        renderField(canvas, neighbors, props.width, props.height);
    });

    return (
        <aside class='surface-container'>
            <canvas
                ref={canvas}
                style={{
                    width: "100%",
                    height: "100%",
                    display: "block"
                }}
            />
        </aside>
    );
}