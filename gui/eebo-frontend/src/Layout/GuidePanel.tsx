
interface Props {
  currentRoute: () => any; // destructure at call site
  onClose: () => void;
}

export const GuidePanel = (props: Props) => {
  return (
    <article class="helpContainer border no-round tiny-padding right surface-container-high large-elevate border shadow"
      style="z-index:999"
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
