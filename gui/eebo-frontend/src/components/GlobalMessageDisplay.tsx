import { Show } from "solid-js";

interface Params {
  title?: string | null;
  errorMessage?: string | null;
  retry?: () => void
}

export default function GlobalMessageDisplay(params: Params) {
  return (
    <div class="background medium no-padding" style="height: 100%">
      <article class={"padding  absolute center middle no-round extra-margin " + (params.errorMessage ? 'error-container' : 'fill')} >
        <h4>{params.title ?? "Loading database"}</h4>
        <Show when={params.errorMessage} fallback={<progress />}>
          <p>
            <code>{params.errorMessage}</code>
            <Show when={params.retry}>
              <button class="chip tiny no-border" onClick={params.retry}>
                <i>refresh</i>
              </button>
            </Show>
          </p>
        </Show>
      </article>
    </div>
  )
}