import "./AutoComplete.css";
import {
    createMemo,
    createSignal,
    For,
    type JSX,
    Show,
} from "solid-js";

interface AutocompleteProps<T> {
    value: string;
    items: T[];

    getLabel: (item: T) => string;
    onInput: (value: string) => void;
    onSelect: (item: T) => void;

    disabled?: boolean;
    placeholder?: string;
    maxSuggestions?: number;
    renderItem?: (item: T) => JSX.Element;

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

    const suggestions = createMemo(() => {
        const query = props.value.trim().toLocaleLowerCase();

        const matching = props.items.filter((item) => {
            const label = props
                .getLabel(item)
                .toLocaleLowerCase();

            return !query || label.includes(query);
        });

        return matching.slice(
            0,
            props.maxSuggestions ?? 8,
        );
    });

    function input(value: string) {
        props.onInput(value);
        setHighlighted(0);

        setOpen(
            value.trim().length > 0 ||
            !!props.openOnFocus,
        );
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

                setHighlighted(
                    Math.min(
                        highlighted() + 1,
                        items.length - 1,
                    ),
                );
                break;

            case "ArrowUp":
                event.preventDefault();

                setHighlighted(
                    Math.max(
                        highlighted() - 1,
                        0,
                    ),
                );
                break;

            case "Enter": {
                const item = items[highlighted()];

                if (item) {
                    event.preventDefault();
                    select(item);
                }

                break;
            }

            case "Escape":
                setOpen(false);
                break;
        }
    }

    return (
        <>
            <div class={`field label ${ isTitle() ? "suffix title" : "border" }`}>
                <input type="text"
                    value={props.value}
                    placeholder={props.isTitle ? '' : props.placeholder}
                    disabled={props.disabled}
                    autocomplete="off"
                    onInput={(event) => input(event.currentTarget.value)}
                    onFocus={() => {
                        if (props.value.trim() || props.openOnFocus) {
                            setOpen(true);
                        }
                    }}
                    onKeyDown={keydown}
                    onBlur={() => {
                        //llow a suggestion click to complete  before closing the menu.
                        setTimeout(() => setOpen(false), 100);
                    }}
                />

                <label> {props.placeholder ?? ""} </label>

                <Show when={isTitle()}>
                    <i>add</i>
                </Show>
            </div>

            <Show when={open() && suggestions().length > 0}>
                <div class="field border">
                    <div class="field autocomplete-menu">
                        <For each={suggestions()}>
                            {(item, index) => (
                                <button type="button"
                                    classList={{
                                        "no-round": true,
                                        active: index() === highlighted(),
                                    }}
                                    onMouseDown={(event) => event.preventDefault()}
                                    onClick={() => select(item)}
                                >
                                    {props.renderItem ? props.renderItem(item) : props.getLabel(item)} </button>
                            )}
                        </For>
                    </div>
                </div>
            </Show>
        </>
    );
}
