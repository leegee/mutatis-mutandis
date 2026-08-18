import ProjectImport from "~/components/ProjectImport";
import ProjectExport from "~/components/ProjectExport";

export default function ProjectPage() {
  return (
    <>
      <article class="large-padding">
        <h2>Project</h2>

        <p>
          Import or export your research
          map as JSON.
        </p>
      </article>

      <article class="large-padding">
        <h3>Import</h3>

        <ProjectImport />
      </article>

      <article class="large-padding">
        <h3>Export</h3>

        <ProjectExport />
      </article>
    </>
  );
}
