// src/components/Modal/Confirm.tsx

import { Show } from "solid-js";

import Modal from "./Modal";

interface ConfirmProps {
    open: boolean;
    title: string;
    message?: string;

    confirmLabel?: string;
    cancelLabel?: string;

    danger?: boolean;

    onConfirm: () => void | Promise<void>;
    onCancel: () => void;
}

export default function Confirm(
    props: ConfirmProps,
) {
    return (
        <Modal
            open={props.open}
            title={props.title}
            onClose={props.onCancel}
            closeOnBackdrop={false}
        >
            <Show when={props.message}>
                <p>{props.message}</p>
            </Show>

            <nav class="footer">
                <button
                    type="button"
                    class={
                        props.danger
                            ? "error"
                            : undefined
                    }
                    onClick={() =>
                        props.onConfirm()
                    }
                >
                    {props.confirmLabel ??
                        "Confirm"}
                </button>

                <button
                    type="button"
                    class="transparent"
                    onClick={() =>
                        props.onCancel()
                    }
                >
                    {props.cancelLabel ??
                        "Cancel"}
                </button>
            </nav>
        </Modal>
    );
}
