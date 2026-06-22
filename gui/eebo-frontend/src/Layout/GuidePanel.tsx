import { onCleanup, onMount } from "solid-js";

interface Props {
  currentRoute: () => any; // destructure at call site
  onClose: () => void;
}

export const GuidePanel = (props: Props) => {
  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === "Escape") props.onClose();
  }

  onMount(() => window.document.body.addEventListener('keydown', handleKeyDown))
  onCleanup(() => window.document.body.removeEventListener('keydown', handleKeyDown))

  return (
    <article class="helpContainer border no-round tiny-padding right surface-container-high large-elevate border shadow"
      style="z-index:99"
      onKeyDown={handleKeyDown}
    >
      <header class="fixed bottom-margin no-padding">
        <nav class="no-padding no-round  surface">
          <span>
            <button class="chip circle" onClick={props.onClose}><i>close</i></button>
            <span class="tooltip bottom">Close</span>
          </span>
          <h2 class="max no-padding noo-margin">{props.currentRoute().label}</h2>
        </nav>
      </header>
      {props.currentRoute().help()}
    </article>
  );
};
