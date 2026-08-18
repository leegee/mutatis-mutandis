import { createEffect, createMemo, createSignal, For, Show } from "solid-js";

import type { Entity } from "~/domain/entity";
import { listEntities } from "~/db/repository";

interface EntityAutocompleteProps {
  value: string;
  onInput: (value: string) => void;
  onSelect: (entity: Entity) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function EntityAutocomplete(
  props: EntityAutocompleteProps,
) {
  const [entities, setEntities] = createSignal<Entity[]>([]);
  const [open, setOpen] = createSignal(false);
  const [highlighted, setHighlighted] = createSignal(0);

  createEffect(() => {
    if (typeof window !== "undefined") {
      listEntities().then(setEntities);
    }
  });

  const suggestions = createMemo(() => {
    const query = props.value.trim().toLocaleLowerCase();

    if (!query) {
      return [];
    }

    return entities()
      .filter((entity) => {
        const label = entity.label.toLocaleLowerCase();

        return (
          label.includes(query) ||
          entity.tags.some((tag) =>
            tag.toLocaleLowerCase().includes(query),
          )
        );
      })
      .slice(0, 8);
  });

  function input(value: string) {
    props.onInput(value);
    setHighlighted(0);
    setOpen(value.trim().length > 0);
  }

  function select(entity: Entity) {
    props.onSelect(entity);
    setOpen(false);
    setHighlighted(0);
  }

  function keydown(event: KeyboardEvent) {
    const items = suggestions();

    if (!open() || items.length === 0) {
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
          Math.max(highlighted() - 1, 0),
        );

        break;

      case "Enter": {
        const entity = items[highlighted()];

        if (entity) {
          event.preventDefault();
          select(entity);
        }

        break;
      }

      case "Escape":
        setOpen(false);
        break;
    }
  }

  return (
    <div class="field label border autocomplete">
      <input
        value={props.value}
        disabled={props.disabled}
        placeholder={props.placeholder ?? "Search entities…"}
        autocomplete="off"
        onInput={(event) =>
          input(event.currentTarget.value)
        }
        onFocus={() => {
          if (props.value.trim()) {
            setOpen(true);
          }
        }}
        onKeyDown={keydown}
      />

      <label>Entity</label>

      <Show when={open() && suggestions().length > 0}>
        <div class="autocomplete-menu surface-container">
          <For each={suggestions()}>
            {(entity, index) => (
              <button
                type="button"
                classList={{
                  active: index() === highlighted(),
                }}
                onMouseDown={(event) => {
                  // Prevent the input from losing focus before
                  // the selection is handled.
                  event.preventDefault();
                }}
                onClick={() => select(entity)}
              >
                <strong>{entity.label}</strong>

                <small>
                  {entity.type}
                </small>
              </button>
            )}
          </For>
        </div>
      </Show>
    </div>
  );
}
