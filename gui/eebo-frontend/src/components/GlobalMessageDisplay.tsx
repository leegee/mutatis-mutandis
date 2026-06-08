import { Show } from "solid-js";

interface Params {
  title?: string | null;
  errorMessage?: string | null;
}

export default function GlobalMessageDisplay(params: Params) {
  return (
    <div class="background medium no-padding" style="height: 100%">
      <article class={"padding  absolute center middle no-round extra-margin " + (params.errorMessage ? 'error-container' : 'fill')} >
        <h4>{params.title ?? "Loading database"}</h4>
        <Show when={!params.errorMessage} fallback={<p>{params.errorMessage}</p>}>
          <progress />
        </Show>
      </article>
    </div>
  )
}