interface EntityImportFieldProps {
	value: string;
	exists: boolean;
	label: string;
	onInput: (value: string) => void;
}

export default function EntityImportField(props: EntityImportFieldProps) {
	return (
		<div class="field label border" classList={{ new: !props.exists }}>
			<input
				type="text"
				value={props.value}
				placeholder={props.label}
				onInput={(event) => props.onInput(event.currentTarget.value)}
			/>

			<label>{props.label}</label>

			<output classList={{ new: !props.exists }}>{props.exists ? "Existing entity" : "Will create entity"}</output>
		</div>
	);
}