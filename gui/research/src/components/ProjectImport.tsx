import { createSignal } from "solid-js";

import type { ResearchProject } from "~/domain/project";
import { importProject } from "~/db/repository";

interface ProjectImportProps {
  onImported?: () => void | Promise<void>;
}

export default function ProjectImport(
  props: ProjectImportProps,
) {
  const [error, setError] = createSignal<string>();
  const [importing, setImporting] = createSignal(false);

  async function handleFile(
    event: Event,
  ) {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];

    if (!file) {
      return;
    }

    setError(undefined);
    setImporting(true);

    try {
      const text = await file.text();
      const project = JSON.parse(text) as ResearchProject;

      validateProject(project);

      await importProject(project);

      await props.onImported?.();

      input.value = "";
    } catch (error) {
      setError(
        error instanceof Error
          ? error.message
          : "Unable to import project.",
      );
    } finally {
      setImporting(false);
    }
  }

  return (
    <div>
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

      {error() && (
        <p role="alert">
          {error()}
        </p>
      )}
    </div>
  );
}

function validateProject(
  project: ResearchProject,
): void {
  if (!project || typeof project !== "object") {
    throw new Error("Invalid project file.");
  }

  if (project.version !== 1) {
    throw new Error(
      `Unsupported project version: ${ project.version }`,
    );
  }

  if (!Array.isArray(project.entities)) {
    throw new Error(
      "Project entities must be an array.",
    );
  }

  if (!Array.isArray(project.relations)) {
    throw new Error(
      "Project relations must be an array.",
    );
  }

  const entityIds = new Set(
    project.entities.map((entity) => entity.id),
  );

  for (const relation of project.relations) {
    if (!entityIds.has(relation.sourceId)) {
      throw new Error(
        `Relation references missing entity: ${ relation.sourceId }`,
      );
    }

    if (!entityIds.has(relation.targetId)) {
      throw new Error(
        `Relation references missing entity: ${ relation.targetId }`,
      );
    }
  }
}
