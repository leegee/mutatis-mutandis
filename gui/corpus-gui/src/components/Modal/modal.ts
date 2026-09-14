import { createSignal, type JSX } from "solid-js";

export type ModalRequest =
	| {
			kind: "alert";
			title?: string;
			message: string;
			resolve: () => void;
			style?: string;
	  }
	| {
			kind: "confirm";
			title?: string;
			message: string;
			resolve: (value: boolean) => void;
			style?: string;
	  }
	| {
			kind: "prompt";
			title?: string;
			message: string;
			defaultValue?: string;
			resolve: (value: string | null) => void;
			style?: string;
	  }
	| {
			kind: "custom";
			title?: string;
			content: (close: () => void) => JSX.Element;
			resolve: () => void;
			style?: string;
	  };

const [current, setCurrent] = createSignal<ModalRequest>();

export function modalState() {
	return current;
}

export function showAlert(
	message: string,
	title?: string,
	style?: string
): Promise<void> {
	return new Promise((resolve) => {
		setCurrent({
			kind: "alert",
			title,
			style,
			message,
			resolve: () => {
				setCurrent(undefined);
				resolve();
			},
		});
	});
}

export function showConfirm(
	message: string,
	title?: string,
	style?: string,):
	Promise<boolean> {
	return new Promise((resolve) => {
		setCurrent({
			kind: "confirm",
			title,
			style,
			message,
			resolve: (value) => {
				setCurrent(undefined);
				resolve(value);
			},
		});
	});
}

export function showPrompt(
	message: string,
	defaultValue?: string,
	title?: string,
	style?: string,
): Promise<string | null> {
	return new Promise((resolve) => {
		setCurrent({
			kind: "prompt",
			title,
			style,
			message,
			defaultValue,
			resolve: (value) => {
				setCurrent(undefined);
				resolve(value);
			},
		});
	});
}

export function showCustom(
	content: (close: () => void) => JSX.Element,
	title?: string,
	style?: string,
): Promise<void> {
	return new Promise((resolve) => {
		setCurrent({
			kind: "custom",
			title,
			content,
			style,
			resolve: () => {
				setCurrent(undefined);
				resolve();
			},
		});
	});
}
