interface Props {
  err: Error;
  reset: () => void;
}

export default function AppError(props: Props) {
  const message = props.err?.message ?? String(props.err);

  const isChunkError = message.startsWith(
    "Failed to fetch dynamically imported module",
  );

  return (
    <article class="small-round padding border medium no-padding">
      <div class="padding error absolute center middle">
        <h3>Error</h3>

        {isChunkError ? (
          <p>App update required. Please refresh.</p>
        ) : (
          <pre style="background: transparent; color: black">{message}</pre>
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
