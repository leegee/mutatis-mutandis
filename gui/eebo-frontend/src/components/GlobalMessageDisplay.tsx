import { Show } from "solid-js";

interface Params {
  title?: string | null;
  errorMessage?: string | null;
}

export default function GlobalMessageDisplay(params: Params) {
  return (
    <article class="small-round padding border medium no-padding" style="height: 100%">
      <div class="padding fill absolute center middle">
        <h4>{params.title ?? "Loading database"}</h4>
        <Show when={!params.errorMessage}>
          <progress />
        </Show>
      </div>
    </article>
  )
}