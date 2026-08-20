import { createEffect, createMemo, createSignal } from "solid-js";
import { listEntities } from "~/db/respository";
import type { Entity } from "~/domain/entity";
import Autocomplete from "./AutoComplete";

interface EntityAutocompleteProps {
  value: string;
  onInput: (value: string) => void;
  onSelect: (entity: Entity) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function EntityAutocomplete(props: EntityAutocompleteProps) {
  const [entities, setEntities] = createSignal<Entity[]>([]);

  createEffect(() => {
    listEntities().then(setEntities);
  });

  const suggestions = createMemo(() => {
    const query = props.value.trim().toLocaleLowerCase();
    if (!query) return [];

    return entities()
      .filter((entity) => {
        const label = entity.label.toLocaleLowerCase();
        return label.includes(query) || entity.tags.some((tag) => tag.toLocaleLowerCase().includes(query));
      })
      .slice(0, 8);
  });

  return (
    <Autocomplete
      value={props.value}
      items={suggestions()}
      getLabel={(entity) => entity.label}
      onInput={props.onInput}
      onSelect={props.onSelect}
      disabled={props.disabled}
      placeholder={props.placeholder ?? "Entity"}
      renderItem={(entity) => (
        <>
          <strong> {entity.label} </strong>
          <small> {entity.type} </small>
        </>
      )}
    />
  );
}
