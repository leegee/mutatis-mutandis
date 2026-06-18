import { createSignal, For, Show } from "solid-js";

type Props = {
  options: string[];
  selected: string[];
  onChange: (val: string[]) => void;
  onCreateOption?: (val: string) => Promise<boolean> | boolean;
};

export default function MultiCreatableSelect(props: Props) {
  const [open, setOpen] = createSignal(false);
  const [query, setQuery] = createSignal("");
  const [creating, setCreating] = createSignal(false);

  const filtered = () => {
    const q = query().toLowerCase();
    return props.options.filter(o =>
      o.toLowerCase().includes(q)
    );
  };

  const toggle = (val: string, checked: boolean) => {
    props.onChange(
      checked
        ? [...props.selected, val]
        : props.selected.filter(v => v !== val)
    );
  };

  const createValue = async (val: string) => {
    if (!props.onCreateOption) return;

    setCreating(true);
    try {
      const ok = await props.onCreateOption(val);
      if (ok) {
        props.onChange([...props.selected, val]);
        setQuery("");
      }
    } finally {
      setCreating(false);
    }
  };

  const displayLabel = () => {
    const len = props.selected.length;
    if (len === 0) return "None selected";
    if (len === 1) return props.selected[0];
    return `${ len } selected`;
  };

  return (
    <div class="field middle-align poo" style="min-width: 15em">
      {/* Trigger */}
      <button
        class="border no-round"
        style="width:100%"
        onClick={() => setOpen(v => !v)}
      >
        <span>{displayLabel()}</span>
        <i>arrow_drop_down</i>
      </button>

      {/* Dropdown */}
      <Show when={open()}>
        <menu id="concept-menu" class="no-round ">

          {/* INPUT FILTER */}
          <li class="padding">
            <div class="field">
              <input
                class="input"
                placeholder="Filter..."
                value={query()}
                onInput={(e) => setQuery(e.currentTarget.value)}
              />
            </div>
          </li>

          {/* OPTIONS */}
          <For each={filtered()}>
            {(c) => (
              <li>
                <label class="checkbox small">
                  <input
                    type="checkbox"
                    checked={props.selected.includes(c)}
                    onChange={(e) =>
                      toggle(c, e.currentTarget.checked)
                    }
                  />
                  <span>{c}</span>
                </label>
              </li>
            )}
          </For>

          {/* CREATE NEW */}
          <Show when={query().trim() && !filtered().includes(query())}>
            <li>
              <label class="checkbox small hover">
                <input
                  type="checkbox"
                  onChange={() => createValue(query())}
                  disabled={creating()}
                />
                <span>
                  <Show
                    when={!creating()}
                    fallback={<progress />}
                  >
                    Create "{query()}"
                  </Show>
                </span>
              </label>
            </li>
          </Show>

        </menu>
      </Show>
    </div>
  );
}
