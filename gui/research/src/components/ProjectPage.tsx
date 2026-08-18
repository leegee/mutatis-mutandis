import { createSignal } from "solid-js";

import ProjectImport from "~/components/ProjectImport";

export default function ProjectPage() {
  const [message, setMessage] = createSignal("");

  async function handleImported() {
    setMessage("Project imported successfully.");
  }

  return (
    <article>
      <section class="large-padding">
        <h2>Import project</h2>

        <ProjectImport />

        {message() && (
          <p>{message()}</p>
        )}
      </section>

      <section class="large-padding">
        <h2>Export project</h2>

        <p>
          Export the current research map as
          JSON.
        </p>

        <button disabled>
          Export project
        </button>
      </section>
    </article>
  );
}
