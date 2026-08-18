import { exportProject } from "~/db/repository";

export default function ProjectExport() {
    async function exportData() {
        const project = await exportProject();

        const json = JSON.stringify(
            project,
            null,
            2,
        );

        const blob = new Blob(
            [json],
            { type: "application/json" },
        );

        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");

        anchor.href = url;
        anchor.download = "research-map.json";

        anchor.click();

        URL.revokeObjectURL(url);
    }

    return (
        <div class="field">
            <label>
                <button onClick={exportData}>
                    Export project
                </button>
            </label>
        </div>
    );
}
