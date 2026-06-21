
interface Props {
  currentRoute: () => any; // destructure at call site
}

export const HelpPanel = (props: Props) => {
  return (
    <article class="helpContainer border no-round large-padding right surface-container-highest large-elevate border"
      style="z-index:999"
    >
      <header class="fixed">
        <nav>
          <button class="small border no-margin no-padding circle" onClick={() => props.currentRoute().setOpenHelp(false)}><i>close</i></button>
          <h2 class="max">{props.currentRoute().label}</h2>
        </nav>
      </header>
      {props.currentRoute().help()}
    </article>
  );
};
