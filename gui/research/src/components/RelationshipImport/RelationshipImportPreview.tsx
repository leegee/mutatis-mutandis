import { For, Show } from "solid-js";
import EntityImportField from "../EntityImportField";
import RelationAutoComplete from "../RelationAutoComplete";
import type { ImportRow, InputField } from "./RelationshipImport.types";

export interface RelationshipImportPreviewProps {
	rows: ImportRow[];
	validRows: ImportRow[];
	rowsToCreate: ImportRow[];
	error: string | undefined;
	saving: boolean;
	updateRow: (index: number, field: InputField, value: string) => void;
	onCancel: () => void;
	onBack: () => void;
	onSubmit: () => void;
}


function rowIsComplete(row: ImportRow): boolean {
	return !row.parseError && row.sourceExists && row.targetExists && row.relationTypeValid && row.relationExists;
}

export function rowIsValid(row: ImportRow): boolean {
	return (
		!row.parseError &&
		row.source.trim().length > 0 &&
		row.type.trim().length > 0 &&
		row.target.trim().length > 0 &&
		row.relationTypeValid
	);
}

function RelationshipRowStatus(props: { row: ImportRow }) {
	return (
		<div class="s1 relationship-import-status">
			<Show
				when={rowIsComplete(props.row)}
				fallback={
					<Show when={rowIsValid(props.row)}>
						<label class="checkbox">
							<input type="checkbox" checked disabled />
							<span />
						</label>
					</Show>
				}
			>
				<span class="relationship-import-ok" aria-description="Relationship already exists">
					✓
				</span>
			</Show>
		</div>
	);
}


export function RelationshipImportPreview(props: RelationshipImportPreviewProps) {
	return (
		<div class="relationship-import-preview">
			<div class="relationship-import-grid">
				<For each={props.rows}>
					{(row, index) => (
						<div class="row relationship-import-row">
							<div class="s4">
								<EntityImportField
									value={row.source}
									exists={row.sourceExists}
									label="Subject Entity"
									onInput={(value) => props.updateRow(index(), "source", value)}
								/>
							</div>

							<div class="s3">
								<RelationAutoComplete
									value={row.type}
									onInput={(value) => props.updateRow(index(), "type", value)}
									onSelect={(type) => props.updateRow(index(), "type", type)}
									outputField={
										row.relationTypeValid ? (
											<output>OK</output>
										) : (
											<output class="invalid-input">Unknown relationship type</output>
										)
									}
								/>
							</div>

							<div class="s4">
								<EntityImportField
									value={row.target}
									exists={row.targetExists}
									label="Object Entity"
									onInput={(value) => props.updateRow(index(), "target", value)}
								/>
							</div>

							<RelationshipRowStatus row={row} />
						</div>
					)}
				</For>
			</div>

			<div class="relationship-import-summary">
				<span>{props.validRows.length} valid</span>

				<span class="left-padding right-padding">{props.rowsToCreate.length} to create</span>

				<span>{props.rows.length - props.validRows.length} invalid</span>
			</div>

			<Show when={props.error}>
				<div class="relationship-import-error error-container">{props.error}</div>
			</Show>

			<nav class="relationship-import-actions footer">
				<button type="button" class="transparent" onClick={props.onCancel} disabled={props.saving}>
					Cancel
				</button>

				<button type="button" class="transparent" onClick={props.onBack} disabled={props.saving}>
					Back
				</button>

				<button type="button" onClick={props.onSubmit} disabled={props.saving || props.rows.length === 0}>
					{props.saving ? "Adding…" : `Add ${ props.rowsToCreate.length }`}
				</button>
			</nav>
		</div>
	);
}
