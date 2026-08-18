// src/components/Modal/Prompt.tsx

import {
    createEffect,
    createSignal,
    Show,
} from "solid-js";

import Modal from "./Modal";

interface PromptProps {
    open: boolean;
    title: string;
    message?: string;
    value?: string;
    placeholder?: string;

    confirmLabel?: string;
    cancelLabel?: string;

    onConfirm: (
        value: string,
    ) => void | Promise<void>;

    onCancel: () => void;
}

export default function Prompt(
    props: PromptProps,
) {
    const [value, setValue] = createSignal(
        props.value ?? "",
    );

    const [saving, setSaving] =
        createSignal(false);

    let input!: HTMLInputElement;

    createEffect(() => {
        if (!props.open) {
            return;
        }

        setValue(props.value ?? "");

        queueMicrotask(() => {
            input?.focus();
            input?.select();
        });
    });

    async function submit(
        event: SubmitEvent,
    ) {
        event.preventDefault();

        if (saving()) {
            return;
        }

        const result = value().trim();

        if (!result) {
            return;
        }

        setSaving(true);

        try {
            await props.onConfirm(result);
        } finally {
            setSaving(false);
        }
    }

    return (
        <Modal
            open={props.open}
            title={props.title}
            onClose={props.onCancel}
            closeOnBackdrop={false}
        >
            <form onSubmit={submit}>
                <Show when={props.message}>
                    <p>{props.message}</p>
                </Show>

                <div class="field label border">
                    <input
                        ref={input}
                        value={value()}
                        placeholder={props.placeholder}
                        disabled={saving()}
                        onInput={(event) =>
                            setValue(
                                event.currentTarget.value,
                            )
                        }
                    />

                    <label>
                        {props.placeholder ?? "Value"}
                    </label>
                </div>

                <nav class="footer">
                    <button
                        type="submit"
                        disabled={
                            saving() ||
                            !value().trim()
                        }
                    >
                        {saving()
                            ? "Saving…"
                            : props.confirmLabel ??
                            "OK"}
                    </button>

                    <button
                        type="button"
                        class="transparent"
                        disabled={saving()}
                        onClick={() =>
                            props.onCancel()
                        }
                    >
                        {props.cancelLabel ??
                            "Cancel"}
                    </button>
                </nav>
            </form>
        </Modal>
    );
}

