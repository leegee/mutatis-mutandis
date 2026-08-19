import { createMemo, createSignal, For, type JSX, Show } from "solid-js";

import "./AutoComplete.css";

interface AutocompleteProps<T> {
    value: string;
    items: T[];

    getLabel: (item: T) => string;
    onInput: (value: string) => void;
    onSelect: (item: T) => void;

    isTitle: boolean;
    disabled?: boolean;
    placeholder?: string;
    maxSuggestions?: number;
    renderItem?: (item: T) => JSX.Element;
}

export default function Autocomplete<T>(props: AutocompleteProps<T>) {
    const [open, setOpen] = createSignal(false);
    const [highlighted, setHighlighted] = createSignal(0);

    const isTitle = createMemo(() => props.isTitle);

    const suggestions = createMemo(() => {
        const query = props.value.trim().toLocaleLowerCase();
        if (!query) return [];

        return props.items
            .filter((item) => props.getLabel(item).toLocaleLowerCase().includes(query))
            .slice(0, props.maxSuggestions ?? 8);
    });

    function input(value: string) {
        props.onInput(value);
        setHighlighted(0);
        setOpen(value.trim().length > 0);
    }

    function select(item: T) {
        props.onSelect(item);
        setOpen(false);
        setHighlighted(0);
    }

    function keydown(event: KeyboardEvent) {
        if (!open()) return;
        const items = suggestions();
        if (items.length === 0) return;

        switch (event.key) {
            case "ArrowDown":
                event.preventDefault();
                setHighlighted(Math.min(highlighted() + 1, items.length - 1));
                break;

            case "ArrowUp":
                event.preventDefault();
                setHighlighted(Math.max(highlighted() - 1, 0));
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
            <div class={`field label ${ isTitle() ? 'suffix title' : 'field border' }`}>
                <input type="text"
                    value={props.value}
                    disabled={props.disabled}
                    autocomplete="off"
                    onInput={(event) => input(event.currentTarget.value)}
                    onFocus={() => {
                        if (props.value.trim()) {
                            setOpen(true);
                        }
                    }}
                    onKeyDown={keydown}
                />

                <label> {props.placeholder ?? "Search"} </label>
                <Show when={isTitle()}>
                    <i>add</i>
                </Show>
            </div>

            <Show when={open() && suggestions().length > 0} >
                <div class="field border">
                    <div class="field autocomplete-menu">
                        <For each={suggestions()}>
                            {(item, index) => (
                                <button
                                    type="button"
                                    classList={{ active: index() === highlighted() }}
                                    onMouseDown={(event) => event.preventDefault()}
                                    onClick={() => select(item)}
                                >
                                    {props.renderItem
                                        ? props.renderItem(item)
                                        : props.getLabel(item)}
                                </button>
                            )}
                        </For>
                    </div>
                </div>
            </Show>
        </>
    );
}
