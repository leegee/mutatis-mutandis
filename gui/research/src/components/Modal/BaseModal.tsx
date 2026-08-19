import {
    createEffect,
    type JSX,
    onCleanup,
    Show,
} from "solid-js";

import { Portal } from "solid-js/web";

import "./modal.css";

interface ModalProps {
    open: boolean;
    title?: string;
    children: JSX.Element;
    onClose?: () => void;
    closeOnBackdrop?: boolean;
}

export default function BaseModal(props: ModalProps) {
    createEffect(() => {
        if (!props.open) {
            return;
        }

        const previousOverflow = document.body.style.overflow;
        document.body.style.overflow = "hidden";

        function handleKeyDown(event: KeyboardEvent) {
            if (event.key === "Escape") {
                props.onClose?.();
            }
        }

        document.addEventListener("keydown", handleKeyDown);

        onCleanup(() => {
            document.body.style.overflow = previousOverflow;
            document.removeEventListener(
                "keydown",
                handleKeyDown,
            );
        });
    });

    function handleBackdropClick(event: MouseEvent) {
        if (
            props.closeOnBackdrop !== false &&
            event.target === event.currentTarget
        ) {
            props.onClose?.();
        }
    }

    return (
        <Show when={props.open}>
            <Portal>
                <div
                    class="modal-backdrop"
                    role="presentation"
                    onClick={handleBackdropClick}
                >
                    <div
                        class="modal"
                        role="dialog"
                        aria-modal="true"
                        aria-label={props.title}
                    >
                        <Show when={props.title}>
                            <header
                                class="fixed surface-container-high padding"
                                style="top:0"
                            >
                                <nav>
                                    <h2 class="max">
                                        {props.title}
                                    </h2>

                                    <Show when={props.onClose}>
                                        <button
                                            class="circle transparent"
                                            type="button"
                                            aria-label="Close"
                                            onClick={() =>
                                                props.onClose?.()
                                            }
                                        >
                                            <i>close</i>
                                        </button>
                                    </Show>
                                </nav>
                            </header>
                        </Show>

                        <div class="modal-content">
                            {props.children}
                        </div>
                    </div>
                </div>
            </Portal>
        </Show>
    );
}
