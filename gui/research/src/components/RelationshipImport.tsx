import { createMemo, createSignal, Show } from "solid-js";

import { importRelationships, listEntities, listRelations } from "~/db/respository";
import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";
import { type RelationType, relationTypes } from "~/domain/relation";
import { useModal } from "./Modal";

import "./RelationshipImports.css";
import type { ImportRow, InputField, ParsedRow } from "./RelationshipImport/RelationshipImport.types";
import { RelationshipImportPreview, rowIsValid } from "./RelationshipImport/RelationshipImportPreview";

interface RelationshipImportProps {
	close: () => void;
}

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

function validateRow(row: ParsedRow, entities: Entity[], relations: Relation[]): ImportRow {
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
}

function resolveRows(parsed: ParsedRow[], entities: Entity[], relations: Relation[]): ImportRow[] {
	return parsed.map((row) => validateRow(row, entities, relations));
}


export function RelationshipImport(props: RelationshipImportProps) {
	const [text, setText] = createSignal("");
	const [rows, setRows] = createSignal<ImportRow[]>([]);
	const [preview, setPreview] = createSignal(false);
	const [saving, setSaving] = createSignal(false);
	const [error, setError] = createSignal<string>();

	// Loaded when the text is parsed, then used for
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
		const current = rows()[index];

		if (!current) return;

		const updated = validateRow(
			{
				...current,
				[field]: value,
				parseError: undefined,
			},
			entities(),
			relations(),
		);

		const next = [...rows()];
		next[index] = updated;

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
		<div class="relationship-import">
			<Show
				when={!preview()}
				fallback={
					<RelationshipImportPreview
						rows={rows()}
						validRows={validRows()}
						rowsToCreate={rowsToCreate()}
						error={error()}
						saving={saving()}
						updateRow={updateRow}
						onCancel={props.close}
						onBack={goBack}
						onSubmit={submit}
					/>
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

					<output class="small-text medium-opacity top-margin bottom-margin">
						aelf --&gt; express --&gt; whiteness
						<br />
						aelf | associated-with | purity
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
		<button type="button" class="transparent no-padding responsive left-align" onClick={handleImportRelationships}>
			Paste
		</button>
	);
}

async function handleImportRelationships() {
	const modal = useModal();

	await modal((close) => <RelationshipImport close={close} />, "Add Items", "min-width: 60rem");
}
