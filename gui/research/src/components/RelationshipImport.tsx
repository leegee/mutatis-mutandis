import { createMemo, createSignal, For, Show } from "solid-js";

import { importRelationships, listEntities, listRelations } from "~/db/respository";
import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";
import { type RelationType, relationTypes } from "~/domain/relation";
import { useModal } from "./Modal";
import RelationAutoComplete from "./RelationAutoComplete";

interface RelationshipImportProps {
	close: () => void;
}

interface ParsedRow {
	source: string;
	type: string;
	target: string;
	parseError?: string;
}

interface ImportRow extends ParsedRow {
	sourceExists: boolean;
	targetExists: boolean;
	relationTypeValid: boolean;
	relationExists: boolean;
}

type InputField = "source" | "type" | "target";

/*
 * Supported forms:
 *
 *   white men --> express --> whiteness
 *   white men | express | whiteness
 *   white men -> express > whiteness
 *
 * The delimiters are deliberately treated as syntax.
 */
function parseLine(line: string): ParsedRow | undefined {
	const trimmed = line.trim();

	if (!trimmed) return undefined;

	const match = trimmed.match(/^\s*(.*?)\s*(?:\|+|--+>|->|>)\s*(.*?)\s*(?:\|+|--+>|->|>)\s*(.*?)\s*$/);

	if (!match) {
		return {
			source: "",
			type: "",
			target: "",
			parseError: "Cannot parse this line.",
		};
	}

	const [, source, type, target] = match;

	if (!source?.trim() || !type?.trim() || !target?.trim()) {
		return {
			source: source?.trim() ?? "",
			type: type?.trim() ?? "",
			target: target?.trim() ?? "",
			parseError: "Source, relationship and target are all required.",
		};
	}

	return {
		source: source.trim(),
		type: type.trim(),
		target: target.trim(),
	};
}

function parseInput(value: string): ParsedRow[] {
	return value
		.split(/\r?\n/)
		.map(parseLine)
		.filter((row): row is ParsedRow => row !== undefined);
}

function findEntity(entities: Entity[], value: string): Entity | undefined {
	const wanted = value.trim();
	return entities.find((entity) => entity.label === wanted || entity.aliases.includes(wanted));
}

function resolveRows(parsed: ParsedRow[], entities: Entity[], relations: Relation[]): ImportRow[] {
	return parsed.map((row) => {
		if (row.parseError) {
			return {
				...row,
				sourceExists: false,
				targetExists: false,
				relationTypeValid: false,
				relationExists: false,
			};
		}

		const source = findEntity(entities, row.source);
		const target = findEntity(entities, row.target);

		const relationTypeValid = relationTypes.includes(row.type as RelationType);

		const relationExists =
			!!source &&
			!!target &&
			relationTypeValid &&
			relations.some(
				(relation) => relation.sourceId === source.id && relation.targetId === target.id && relation.type === row.type,
			);

		return {
			...row,
			sourceExists: !!source,
			targetExists: !!target,
			relationTypeValid,
			relationExists,
		};
	});
}

function rowIsComplete(row: ImportRow): boolean {
	return !row.parseError && row.sourceExists && row.targetExists && row.relationTypeValid && row.relationExists;
}

function rowIsValid(row: ImportRow): boolean {
	return (
		!row.parseError &&
		row.source.trim().length > 0 &&
		row.type.trim().length > 0 &&
		row.target.trim().length > 0 &&
		row.relationTypeValid
	);
}

export function RelationshipImport(props: RelationshipImportProps) {
	const [text, setText] = createSignal("");
	const [rows, setRows] = createSignal<ImportRow[]>([]);
	const [preview, setPreview] = createSignal(false);
	const [saving, setSaving] = createSignal(false);
	const [error, setError] = createSignal<string>();

	// These are loaded once when the text is parsed, then used for
	// live validation while the user edits the preview.
	const [entities, setEntities] = createSignal<Entity[]>([]);
	const [relations, setRelations] = createSignal<Relation[]>([]);

	const validRows = createMemo(() => rows().filter(rowIsValid));

	const rowsToCreate = createMemo(() => rows().filter((row) => rowIsValid(row) && !row.relationExists));

	const canSubmit = createMemo(() => {
		const current = rows();

		return !saving() && current.length > 0 && current.every(rowIsValid);
	});

	async function parse() {
		setError(undefined);

		const parsed = parseInput(text());

		if (parsed.length === 0) {
			setError("Enter at least one relationship.");
			return;
		}

		const [loadedEntities, loadedRelations] = await Promise.all([listEntities(), listRelations()]);

		setEntities(loadedEntities);
		setRelations(loadedRelations);
		setRows(resolveRows(parsed, loadedEntities, loadedRelations));
		setPreview(true);
	}

	function updateRow(index: number, field: InputField, value: string) {
		const next = [...rows()];

		const row = next[index];
		if (!row) return;

		next[index] = {
			...row,
			[field]: value,
			parseError: undefined,
		};

		const updated = next[index];
		if (!updated) return;

		const source = findEntity(entities(), updated.source);
		const target = findEntity(entities(), updated.target);
		const relationTypeValid = relationTypes.includes(updated.type as RelationType);

		const relationExists =
			!!source &&
			!!target &&
			relationTypeValid &&
			relations().some(
				(relation) =>
					relation.sourceId === source.id && relation.targetId === target.id && relation.type === updated.type,
			);

		next[index] = {
			...updated,
			sourceExists: !!source,
			targetExists: !!target,
			relationTypeValid,
			relationExists,
		};

		setRows(next);
		setError(undefined);
	}

	function goBack() {
		setPreview(false);
		setError(undefined);
	}

	async function submit() {
		if (!canSubmit()) return;

		setSaving(true);
		setError(undefined);

		try {
			await importRelationships(
				rowsToCreate().map((row) => ({
					source: row.source.trim(),
					type: row.type.trim() as RelationType,
					target: row.target.trim(),
				})),
			);

			props.close();
		} catch (cause) {
			console.error(cause);
			setError(cause instanceof Error ? cause.message : "Could not add the relationships.");
		} finally {
			setSaving(false);
		}
	}

	return (
		<div class="relationship-import padding">
			<Show
				when={!preview()}
				fallback={
					<>
						<div class="relationship-import-grid">
							<For each={rows()}>
								{(row, index) => (
									<div class="row relationship-import-row">
										{/* Source */}
										<div class="s4">
											<div class={`field label border ${ !row.sourceExists ? "new" : "" }`}>
												<input
													type="text"
													value={row.source}
													placeholder="Subject Entity"
													onInput={(event) => updateRow(index(), "source", event.currentTarget.value)}
												/>

												<label>Subject Entity</label>

												<output class={!row.sourceExists ? "new" : ""}>
													{row.sourceExists ? "Existing entity" : "Will create entity"}
												</output>
											</div>
										</div>

										{/* Relationship */}
										<div class="s3">
											<RelationAutoComplete
												value={row.type}
												onInput={(value) => updateRow(index(), "type", value)}
												onSelect={(type) => updateRow(index(), "type", type)}
												outputField={!row.relationTypeValid ? <output class="invalid-input">Unknown relationship type</output> : ""}
											/>
										</div>

										{/* Target */}
										<div class="s4">
											<div class={`field label border ${ !row.targetExists ? "new" : "" }`}>
												<input
													type="text"
													value={row.target}
													placeholder="Object Entity"
													onInput={(event) => updateRow(index(), "target", event.currentTarget.value)}
												/>

												<label>Object Entity</label>

												<output class={!row.targetExists ? "new" : ""}>
													{row.targetExists ? "Existing entity" : "Will create entity"}
												</output>
											</div>
										</div>

										{/* Status */}
										<div class="s1 relationship-import-status">
											<Show when={rowIsComplete(row)} fallback={
												<Show when={rowIsValid(row)}>
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
									</div>
								)}
							</For>
						</div>

						<div class="relationship-import-summary top-margin bottom-margin">
							<span>{validRows().length} valid</span>
							<span class="left-padding right-padding">{rowsToCreate().length} to create</span>
							<span>{rows().length - validRows().length} invalid</span>
						</div>

						<Show when={error()}>
							<div class="relationship-import-error error-container">{error()}</div>
						</Show>

						<nav class="relationship-import-actions footer">
							<button type="button" onClick={props.close} disabled={saving()}>
								Cancel
							</button>

							<button type="button" onClick={goBack} disabled={saving()}>
								Back
							</button>

							<button type="button" onClick={submit} disabled={!canSubmit()}>
								{saving() ? "Adding…" : `Add ${ rowsToCreate().length }`}
							</button>
						</nav>
					</>
				}
			>
				<div class="field border label">
					<textarea
						class="relationship-import-input"
						style="min-height: 10em"
						value={text()}
						onInput={(event) => setText(event.currentTarget.value)}
						autofocus
						placeholder="Paste graph code"
					/>
					<label>Paste graph code</label>
					<output class="small-text medium-opacity">
						white men --&gt; express --&gt; whiteness
						<br />
						white men | associated-with | purity
						<br />
						hvítr &gt; expresses &gt; whiteness
					</output>
				</div>

				<Show when={error()}>
					<div class="relationship-import-error">{error()}</div>
				</Show>

				<nav class="footer">
					<button type="button" class="transparent" onClick={props.close}>
						Cancel
					</button>

					<button type="button" onClick={parse} disabled={!text().trim()}>
						Submit
					</button>
				</nav>
			</Show>
		</div>
	);
}

export default function RelationshipImportButton() {
	return (
		<button type="button" class="transparent no-padding" onClick={handleImportRelationships}>
			Paste
		</button>
	);
}

async function handleImportRelationships() {
	const modal = useModal();
	await modal((close) => <RelationshipImport close={close} />, "Add Items", "min-width: 60rem");
}
