/**
 * ContextAside.tsx
 *
 * Shows the token-window text excerpt and a button to open the full document.
 * Rendered at the top of the centre panel whenever an event is active.
 */

import { Show, type Component } from "solid-js";
import type { SqliteEventWithNeighbours } from "../../types";
import { showDocument } from "../../services/documentApi";

interface Props {
  event: () => SqliteEventWithNeighbours;
  windowText: () => string | undefined;
}

const ContextAside: Component<Props> = (props) => (
  <aside class="center-align small-padding border small-round">
    <Show
      when={props.windowText()}
      fallback={
        <div>
          <p>Loading context</p>
          <progress />
        </div>
      }
    >
      {(text) => (
        <>
          <blockquote innerHTML={text()} />
          <button
            class="chip"
            disabled={!props.event().doc_id}
            onClick={() => {
              const { doc_id, token_idx } = props.event();
              if (doc_id) showDocument(doc_id, token_idx);
            }}
          >
            {props.event().doc_id ?? "No document"}
          </button>
        </>
      )}
    </Show>
  </aside>
);

export default ContextAside;
