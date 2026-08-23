import { createSignal } from "solid-js";
import { useAlert, useConfirm } from "~/components/Modal/";
import { importProject } from "~/db/respository";
import { validateProject } from "~/domain/validateProject";

export default function ProjectImport() {
	const [importing, setImporting] = createSignal(false);
	const alert = useAlert();
	const confirm = useConfirm();

	async function handleFile(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;

		setImporting(true);

		const ok = confirm("Loading this file will cause existing data to be lost.")
		if (!ok) return;

		try {
			const text = await file.text();
			const value: unknown = JSON.parse(text);

			const validation = validateProject(value);

			if (!validation.valid) {
				throw new Error(validation.errors.map((error) => `${ error.path }: ${ error.message }`).join("\n"));
			}

			await importProject(value);

			input.value = "";
			await alert(`Import complete - "${ file.name }" was imported successfully.`);
		} catch (error) {
			await alert(`Unable to import project - ${ error instanceof Error ? error.message : "Unable to import project." }`);
		} finally {
			setImporting(false);
		}
	}

	return (
		<>
			<button
				type="button"
				class="small transparent no-padding"
				disabled={importing()}
				onclick={() => document.getElementById("project-import")?.click()}
			>
				{importing() ? "Importing…" : "Import JSON"}
			</button>

			<input
				id="project-import"
				type="file"
				accept="application/json,.json"
				hidden
				onChange={handleFile}
				style="display:none"
			/>
		</>
	);
}
