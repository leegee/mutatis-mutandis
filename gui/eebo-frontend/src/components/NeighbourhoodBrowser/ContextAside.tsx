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
  event: () => Pick<SqliteEventWithNeighbours, "doc_id" | "token_idx">;
  windowText: () => string | null | undefined;
}

const ContextAside: Component<Props> = (props) => (
  <aside class="center-align small-padding border small-round">
    <Show when={props.windowText()}
      fallback={
        <div>
          <p>Loading context</p>
          <progress />
        </div>
      }
    >
      {(text) => (
        <Show when={props.event().doc_id}>
          <blockquote innerHTML={text()} />
          <button class="chip" onClick={() => showDocument(props.event().doc_id, props.event().token_idx)} >
            <span> {props.event().doc_id}</span>
            <i class="small medium-opacity"> open_in_new</i>
          </button>
        </Show>
      )}
    </Show>
  </aside >
);

export default ContextAside;
