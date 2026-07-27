import { For, Show, type Component, createSignal, onMount } from "solid-js";
import type { NodeMeta } from "./types";

interface Props {
  tip: TipData;
}

export interface TipData {
  node: NodeMeta;
  x: number;
  y: number;
}

export const Tooltip: Component<Props> = (p) => {
  let tooltipRef!: HTMLDivElement;

  const [position, setPosition] = createSignal({
    x: p.tip.x + 14,
    y: p.tip.y - 10,
  });

  onMount(() => {
    const rect = tooltipRef.getBoundingClientRect();

    const margin = 25;

    let x = p.tip.x + 14;

    // Keep tooltip inside viewport horizontally
    if (x + rect.width > window.innerWidth - margin) {
      x = window.innerWidth - rect.width - margin;
    }

    // Avoid clipping at the left edge
    if (x < margin) {
      x = margin;
    }

    let y = p.tip.y - 10;

    // Keep tooltip inside viewport vertically
    if (y + rect.height > window.innerHeight - margin) {
      y = window.innerHeight - rect.height - margin;
    }

    // Avoid clipping at the top
    if (y < margin) {
      y = margin;
    }

    setPosition({ x, y });
  });
  return (
    <aside
      ref={tooltipRef}
      class="surface-container-highest border padding large-elevate"
      style={{
        position: "absolute",
        left: `${ position().x }px`,
        top: `${ position().y }px`,
        "max-width": "260px",
        "z-index": "10",
      }}
    >
      <h6>
        {p.tip.node.label}
        {p.tip.node.pubYear && ` · ${ p.tip.node.pubYear }`}
      </h6>

      <Show when={p.tip.node.neighbourTokens?.length}>
        <div
          style={{
            "margin-top": "8px",
            opacity: "0.7",
            "font-size": "0.85rem",
          }}
        >
          semantic field
        </div>

        <For each={p.tip.node.neighbourTokens}>
          {(entry) => (
            <div>
              {entry[0]} × {entry[1]}
            </div>
          )}
        </For>
      </Show>

      <Show when={p.tip.node.docId}>
        <div
          style={{
            "margin-top": "8px",
            opacity: "0.45",
            "font-size": "0.8rem",
          }}
        >
          {p.tip.node.docId}
        </div>
      </Show>
    </aside>
  );
};