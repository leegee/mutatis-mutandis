import { createSignal } from "solid-js";

import { importProject } from "~/db/repository";
import { validateProject } from "~/domain/validateProject";
import { useAlert } from "~/components/Modal/";

interface ProjectImportProps {
}

export default function ProjectImport(
  props: ProjectImportProps,
) {

  const [importing, setImporting] =
    createSignal(false);

  async function handleFile(
    event: Event,
  ) {
    const input = event.currentTarget as HTMLInputElement;

    const file = input.files?.[0];

    if (!file) {
      return;
    }

    setImporting(true);

    try {
      const text = await file.text();
      const value: unknown = JSON.parse(text);

      const validation =
        validateProject(value);

      if (!validation.valid) {
        throw new Error(
          validation.errors
            .map(
              (error) =>
                `${ error.path }: ${ error.message }`,
            )
            .join("\n"),
        );
      }

      await importProject(value);

      input.value = "";

      await alert(`Import complete - "${ file.name }" was imported successfully.`);
    }
    catch (error) {
      await alert(`Unable to import project - ${ error instanceof Error ? error.message : "Unable to import project." }`);
    } finally {
      setImporting(false);
    }
  }

  return (
    <>
      <div class="field">
        <label>
          <button
            class="small secondary"
            type="button"
            disabled={importing()}
            onclick={() =>
              document
                .getElementById("project-import")
                ?.click()
            }
          >
            {importing()
              ? "Importing…"
              : "Import JSON"}
          </button>

          <input
            id="project-import"
            type="file"
            accept="application/json,.json"
            hidden
            onChange={handleFile}
          />
        </label>
      </div>
    </>
  );
}
