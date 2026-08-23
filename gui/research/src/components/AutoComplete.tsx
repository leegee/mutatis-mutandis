import "./AutoComplete.css";
import { createMemo, createSignal, For, type JSX, Show } from "solid-js";


interface AutocompleteProps<T> {
	value: string;
	items: T[];

	getLabel: (item: T) => string;
	onInput: (value: string) => void;
	onSelect: (item: T) => void;

	onEnter?: () => void;
	selectOnEnter?: boolean;

	disabled?: boolean;
	placeholder?: string;
	maxSuggestions?: number;
	renderItem?: (item: T) => JSX.Element;

	outputField?: JSX.Element | string;

	onCreate?: (value: string) => void | Promise<void>;

	// Displays the input as a title with a + icon
	isTitle?: boolean;

	// Show suggestions as soon as the field receives focus, even when the input is empty.
	openOnFocus?: boolean;

	// Clear the input after an item is selected, for multi-value selectors such as entities, relations, tags and aliases.
	clearOnSelect?: boolean;
}

export default function AutoComplete<T>(props: AutocompleteProps<T>) {
	const [open, setOpen] = createSignal(false);
	const [highlighted, setHighlighted] = createSignal(0);
	const isTitle = createMemo(() => props.isTitle);

	const createValue = () => props.value.trim();

	const canCreate = () =>
		!!props.onCreate &&
		createValue().length > 0 &&
		!props.items.some((item) => props.getLabel(item).toLocaleLowerCase() === createValue().toLocaleLowerCase());

	const suggestions = createMemo(() => {
		const query = props.value.trim().toLocaleLowerCase();

		const matching = props.items.filter((item) => {
			const label = props.getLabel(item).toLocaleLowerCase();

			return !query || label.includes(query);
		});

		return matching.slice(0, props.maxSuggestions ?? 8);
	});

	function input(value: string) {
		props.onInput(value);
		setHighlighted(0);

		setOpen(value.trim().length > 0 || !!props.openOnFocus);
	}

	function select(item: T) {
		props.onSelect(item);

		if (props.clearOnSelect) {
			props.onInput("");
		}

		setOpen(false);
		setHighlighted(0);
	}

	function keydown(event: KeyboardEvent) {
		if (event.key === "Enter") {
			const items = suggestions();
			const item = items[highlighted()];

			if (item && props.selectOnEnter !== false) {
				event.preventDefault();
				select(item);
				return;
			}

			props.onEnter?.();
			return;
		}

		if (!open()) {
			return;
		}

		const items = suggestions();

		if (items.length === 0) {
			return;
		}

		switch (event.key) {
			case "ArrowDown":
				event.preventDefault();
				setHighlighted(Math.min(highlighted() + 1, items.length - 1));
				break;

			case "ArrowUp":
				event.preventDefault();
				setHighlighted(Math.max(highlighted() - 1, 0));
				break;

			case "Escape":
				setOpen(false);
				break;
		}
	}

	return (
		<div class="autocomplete padding">
			<div class={`field label ${ isTitle() ? "suffix title" : "border" }`}>
				<input
					type="text"
					value={props.value}
					placeholder={props.isTitle ? "" : props.placeholder}
					disabled={props.disabled}
					autocomplete="off"
					onKeyDown={keydown}
					onBlur={() => { setTimeout(() => setOpen(false), 100); }}
					onInput={(event) => input(event.currentTarget.value)}
					onFocus={() => {
						if (props.value.trim() || props.openOnFocus) {
							setOpen(true);
						}
					}}
				/>
				<label> {props.placeholder} </label>
				<Show when={props.outputField}>
					{props.outputField}
				</Show>
				<Show when={isTitle()}>
					<i>add</i>
				</Show>
			</div>

			<Show when={open() && (suggestions().length > 0 || canCreate())}>
				<div class="autocomplete-menu elevate">
					<div class="field border">
						<For each={suggestions()}>
							{(item, index) => (
								<button
									type="button"
									classList={{
										"no-round": true,
										active: index() === highlighted(),
									}}
									onMouseDown={(event) => event.preventDefault()}
									onClick={() => select(item)}
								>
									{props.renderItem
										? props.renderItem(item)
										: props.getLabel(item)}
								</button>
							)}
						</For>

						<Show when={canCreate()}>
							<button
								type="button"
								class="no-round"
								onMouseDown={(event) => event.preventDefault()}
								onClick={() => {
									const value = createValue();

									if (value) {
										void props.onCreate?.(value);
									}
								}}
							>
								<i>add</i>
								Create "{createValue()}"
							</button>
						</Show>
					</div>
				</div>
			</Show>
		</div>
	);
}
