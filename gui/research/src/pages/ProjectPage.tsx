import ProjectExport from "~/components/Project/ProjectExport";
import ProjectImport from "~/components/Project/ProjectImport";
import ProjectReset from "~/components/Project/ResetProject";

export default function ProjectPage() {
	return (
		<article class="top-level-view">
			<section>
				<h2>Project</h2>
				<div class="field">
					<ProjectImport />
				</div>
				<div class="field">
					<ProjectExport />
				</div>
			</section>

			<section class="warning-container">
				<h2>Reset Project</h2>
				<div class="field">
					<ProjectReset />
				</div>
			</section>
		</article>
	);
}
