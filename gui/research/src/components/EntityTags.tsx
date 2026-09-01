import { createEffect, createSignal, For, Show } from "solid-js";
import {
	addEntityTag,
	listTags,
	removeEntityTag,
} from "~/db/respository";
import type { Entity } from "~/domain/entity";
import AutoComplete from "./AutoComplete";

const no_data_fallback_class = "bottom-padding no-margin center-align";

interface EntityTagsProps {
	entity: Entity;

	onChanged?: (entity: Entity) => void | Promise<void>;
}

export default function EntityTags(props: EntityTagsProps) {
	const [tagInput, setTagInput] = createSignal("");
	const [tags, setTags] = createSignal<string[]>([]);

	// The global vocabulary of available tags.
	createEffect(() => {
		listTags().then(setTags);
	});

	async function handleAddTag(tag: string) {
		const value = tag.trim();
		if (!value) return;

		const updated = await addEntityTag(props.entity, value);

		setTagInput("");

		// The tag may have been newly created, so refresh the vocabulary.
		setTags(await listTags());

		await props.onChanged?.(updated);
	}

	async function handleRemoveTag(tag: string) {
		const updated = await removeEntityTag(props.entity, tag);

		await props.onChanged?.(updated);
	}

	return (
		<section class="surface-container">
			<AutoComplete<string>
				value={tagInput()}
				items={tags()}
				getLabel={(tag) => tag}
				onInput={setTagInput}
				onSelect={handleAddTag}
				onEnter={() => handleAddTag(tagInput())}
				placeholder="Tags"
				isTitle
			/>

			<Show when={props.entity.tags.length > 0} fallback={<p class={no_data_fallback_class}>No tags.</p>} >
				<div class="row wrap tiny-space tiny-margin bottom-margin top-margin">
					<For each={props.entity.tags}>
						{(tag) => (
							<span class="small chip left-padding">
								{tag}

								<button type="button"
									class="transparent small circle no-padding"
									title={`Remove ${ tag }`}
									aria-label={`Remove tag ${ tag }`}
									onClick={() => handleRemoveTag(tag)}
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
