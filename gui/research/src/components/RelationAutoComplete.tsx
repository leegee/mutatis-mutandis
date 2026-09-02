import { createMemo, type JSX } from "solid-js";

import AutoComplete from "~/components/AutoComplete";
import { type RelationType, relationTypes } from "~/domain/relation";

interface RelationAutoCompleteProps {
	value: string;
	onInput: (value: string) => void;
	onSelect: (type: RelationType) => void;
	outputField?: JSX.Element | string;
}

export default function RelationAutoComplete(props: RelationAutoCompleteProps) {
	const items = createMemo(() => [...relationTypes]);

	return (
		<AutoComplete<RelationType>
			value={props.value}
			items={items()}
			getLabel={(type) => type}
			onInput={props.onInput}
			onSelect={props.onSelect}
			placeholder="Relationship"
			openOnFocus
			clearOnSelect={false}
			maxSuggestions={10}
			outputField={props.outputField}
		/>
	);
}
