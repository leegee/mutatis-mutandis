import {
    createEffect,
    createSignal,
    Show,
} from "solid-js";

import BaseModal from "./BaseModal";
import { modalState } from "./modal";

export default function ModalHost() {
    const request = modalState();

    const [value, setValue] = createSignal("");

    createEffect(() => {
        const current = request();

        if (current?.kind === "prompt") {
            setValue(current.defaultValue ?? "");
        } else {
            setValue("");
        }
    });

    function close() {
        const current = request();

        if (!current) {
            return;
        }

        switch (current.kind) {
            case "alert":
                current.resolve();
                break;

            case "confirm":
                current.resolve(false);
                break;

            case "prompt":
                current.resolve(null);
                break;

            // case "custom":
            //     current.resolve();
            //     break;
        }
    }

    return (
        <Show when={request()}>
            {(current) => (
                <BaseModal
                    open={true}
                    title={
                        // current().kind === "custom" ? undefined : current().title
                        current().title
                    }
                    onClose={close}
                >
                    {(() => {
                        const modal = current();

                        switch (modal.kind) {
                            case "alert":
                                return (
                                    <div class="padding">
                                        <p>
                                            {modal.message}
                                        </p>

                                        <nav class="footer">
                                            <button type="button" onClick={() => modal.resolve()} >
                                                OK
                                            </button>
                                        </nav>
                                    </div>
                                );

                            case "confirm":
                                return (
                                    <div class="padding">
                                        <p>
                                            {modal.message}
                                        </p>

                                        <nav class="footer">
                                            <button type="button" class="error" onClick={() => modal.resolve(true)} >
                                                Confirm
                                            </button>

                                            <button type="button" class="transparent" onClick={() => modal.resolve(false)} >
                                                Cancel
                                            </button>
                                        </nav>
                                    </div>
                                );

                            case "prompt":
                                return (
                                    <form
                                        onSubmit={(event) => {
                                            event.preventDefault();
                                            modal.resolve(
                                                value(),
                                            );
                                        }}
                                    >
                                        <div class="padding">
                                            <p>
                                                {modal.message}
                                            </p>

                                            <div class="field label border">
                                                <input
                                                    autofocus
                                                    value={value()}
                                                    onInput={(event) =>
                                                        setValue(
                                                            event
                                                                .currentTarget
                                                                .value,
                                                        )
                                                    }
                                                />

                                                <label>
                                                    {modal.message}
                                                </label>
                                            </div>

                                            <nav class="footer">
                                                <button type="submit">
                                                    OK
                                                </button>

                                                <button type="button" class="transparent" onClick={() => modal.resolve(null)} >
                                                    Cancel
                                                </button>
                                            </nav>
                                        </div>
                                    </form>
                                );

                            // case "custom":
                            //     return modal.content;
                        }
                    })()}
                </BaseModal>
            )}
        </Show>
    );
}