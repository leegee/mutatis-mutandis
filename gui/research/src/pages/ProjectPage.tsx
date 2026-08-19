import ProjectExport from "~/components/ProjectExport";
import ProjectImport from "~/components/ProjectImport";

export default function ProjectPage() {
  return (
    <article class="small page active">
      <section>
        <h2>Project</h2>

        <p>Import or export your research map as JSON.</p>

        <div class="row">
          <ProjectImport />
          <ProjectExport />
        </div>
      </section>
    </article >
  );
}
