interface Props {
  err: Error;
  reset: () => void;
}

export default function AppError(props: Props) {
  const isChunkError = props.err.message.startsWith(
    "Failed to fetch dynamically imported module",
  );

  return (
    <article class="small-round padding border medium no-padding">
      <div class="padding error absolute center middle">
        <h3>Error</h3>
        {isChunkError ? (
          <p>App update required. Please refresh.</p>
        ) : (
          <p>Something went wrong: {props.err.message}</p>
        )}
        <nav>
          <button class="error border" onClick={props.reset}>
            <i>restart_alt</i>
            <span>Retry</span>
          </button>
        </nav>
      </div>
    </article>
  );
}
