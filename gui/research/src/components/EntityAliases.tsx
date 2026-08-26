import { createEffect, createSignal, For, Show } from "solid-js";
import {
	addEntityAlias,
	listAliases,
	removeEntityAlias,
} from "~/db/respository";
import type { Entity } from "~/domain/entity";
import AutoComplete from "./AutoComplete";

const no_data_fallback_class = "bottom-padding no-margin center-align";

interface EntityAliasesProps {
	entity: Entity;

	onChanged?: (entity: Entity) => void | Promise<void>;
}

export default function EntityAliases(props: EntityAliasesProps) {
	const [aliasInput, setAliasInput] = createSignal("");
	const [aliases, setAliases] = createSignal<string[]>([]);

	// The global vocabulary of available aliases.
	createEffect(() => {
		listAliases().then(setAliases);
	});

	async function handleAddAlias(alias: string) {
		const value = alias.trim();
		if (!value) return;

		const updated = await addEntityAlias(props.entity, value);

		setAliasInput("");

		// The alias may have been newly created, so refresh the vocabulary.
		setAliases(await listAliases());

		await props.onChanged?.(updated);
	}

	async function handleRemoveAlias(alias: string) {
		const updated = await removeEntityAlias(props.entity, alias);

		await props.onChanged?.(updated);
	}

	return (
		<section class="surface">
			<AutoComplete<string>
				value={aliasInput()}
				items={aliases()}
				getLabel={(alias) => alias}
				onInput={setAliasInput}
				onEnter={() => handleAddAlias(aliasInput())}
				onSelect={handleAddAlias}
				placeholder="Aliases "
				isTitle
			/>

			<Show
				when={props.entity.aliases.length > 0}
				fallback={
					<p class={no_data_fallback_class}>No aliases.</p>
				}
			>
				<div class="row wrap tiny-space">
					<For each={props.entity.aliases}>
						{(alias) => (
							<span class="small chip left-padding">
								{alias}

								<button
									type="button"
									class="transparent small circle no-padding"
									title={`Remove ${ alias }`}
									aria-label={`Remove alias ${ alias }`}
									onClick={() => handleRemoveAlias(alias)}
								>
									<i class="small">close</i>
								</button>
							</span>
						)}
					</For>
				</div>
			</Show>
		</section>
	);
}
