import ProjectExport from "~/components/ProjectExport";
import ProjectImport from "~/components/ProjectImport";

export default function ProjectPage() {
  return (
    <>
      <article class="large-padding">
        <h2>Project</h2>

        <p>Import or export your research map as JSON.</p>
      </article>

      <article class="large-padding">
        <div class="row">
          <ProjectImport />
          <ProjectExport />
        </div>
      </article>
    </>
  );
}
