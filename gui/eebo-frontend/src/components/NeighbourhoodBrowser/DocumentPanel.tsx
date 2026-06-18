/**
 * DocumentPanel.tsx
 *
 * Right panel: lists documents associated with the focused token or
 * the selected event. Clicking a row previews that document's context window.
 */

import { For, Show, type Component } from "solid-js";

interface DocEntry {
  docId: string;
  year?: number;
  token_idx: number;
}

interface Props {
  docs: () => DocEntry[];
  focusToken: () => string | null;
  rightPanelEvent: () => { doc_id: string; token_idx: number } | null;
  onSelectDoc: (docId: string, tokenIdx: number) => void;
}

const DocumentPanel: Component<Props> = (props) => (
  <aside class="s3 surface-container">
    <div
      class="padding small-text bold"
      style={{ "border-bottom": "1px solid rgba(255,255,255,0.08)" }}
    >
      <Show when={props.focusToken()} fallback="Documents">
        <span>
          Documents for <q>{props.focusToken()}</q>
          <span class="small-text left-padding medium-opacity">
            {props.docs().length}
          </span>
        </span>
      </Show>
    </div>

    <Show
      when={props.docs().length > 0}
      fallback={
        <div class="padding small-opacity small-text">
          Select an event or click a neighbour token
        </div>
      }
    >
      <div style={{ padding: "0.5rem" }}>
        <For each={props.docs()}>
          {({ docId, year, token_idx }) => {
            const isActive = () => props.rightPanelEvent()?.doc_id === docId;

            return (
              <button
                class={`chip small-margin ${ isActive() ? "primary" : "" }`}
                style={{
                  display: "flex",
                  "justify-content": "space-between",
                  width: "calc(100% - 0.5rem)",
                  cursor: "pointer",
                }}
                onClick={() => props.onSelectDoc(docId, token_idx)}
              >
                <span style={{
                  "font-family": "'IBM Plex Mono', monospace",
                  "font-size": "0.78rem",
                  overflow: "hidden",
                  "text-overflow": "ellipsis",
                }}>
                  {docId}
                </span>
                <Show when={year !== undefined}>
                  <span
                    class="small-text medium-opacity"
                    style={{ "flex-shrink": "0", "padding-left": "0.4rem" }}
                  >
                    {year}
                  </span>
                </Show>
              </button>
            );
          }}
        </For>
      </div>
    </Show>
  </aside>
);

export default DocumentPanel;
