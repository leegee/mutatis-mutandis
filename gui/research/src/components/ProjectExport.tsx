import { exportProject } from "~/db/respository";

export default function ProjectExport() {
	async function exportData() {
		const project = await exportProject();

		const json = JSON.stringify(project, null, 2);

		const blob = new Blob([json], { type: "application/json" });

		const url = URL.createObjectURL(blob);
		const anchor = document.createElement("a");

		anchor.href = url;
		anchor.download = "research-map.json";

		anchor.click();

		URL.revokeObjectURL(url);
	}

	return (
		<button type="button" class="small transparent no-padding" onClick={exportData}>
			Export project
		</button>
	);
}
