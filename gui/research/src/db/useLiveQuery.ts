import type { Observable } from "dexie";
import { createSignal, onCleanup, onMount } from "solid-js";

export function useLiveQuery<T>(observable: Observable<T>, initial: T) {
	const [value, setValue] = createSignal<T>(initial);
	const [loading, setLoading] = createSignal(true);

	onMount(() => {
		const subscription = observable.subscribe({
			next: (nextValue) => {
				setValue(() => nextValue);
				setLoading(false);
			},

			error: (error) => {
				console.error(error);
				setLoading(false);
			},
		});

		onCleanup(() => {
			subscription.unsubscribe();
		});
	});

	return {
		value,
		loading,
	};
}
