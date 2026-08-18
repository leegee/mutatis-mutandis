import { createSignal } from "solid-js";

import { importProject } from "~/db/repository";
import { validateProject } from "~/domain/validateProject";
import Alert from "~/components/Modal/Alert";

interface ProjectImportProps {
}

export default function ProjectImport(
  props: ProjectImportProps,
) {
  const [alert, setAlert] =
    createSignal<{
      title: string;
      message: string;
    }>();

  const [importing, setImporting] =
    createSignal(false);

  async function handleFile(
    event: Event,
  ) {
    const input =
      event.currentTarget as HTMLInputElement;

    const file = input.files?.[0];

    if (!file) {
      return;
    }

    setAlert(undefined);
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

      setAlert({
        title: "Import complete",
        message: `"${ file.name }" was imported successfully.`,
      });
    } catch (error) {
      setAlert({
        title: "Unable to import project",
        message:
          error instanceof Error
            ? error.message
            : "Unable to import project.",
      });
    } finally {
      setImporting(false);
    }
  }

  function closeAlert() {
    setAlert(undefined);
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

      <Alert
        open={!!alert()}
        title={alert()?.title ?? ""}
        message={alert()?.message ?? ""}
        onClose={closeAlert}
      />
    </>
  );
}
