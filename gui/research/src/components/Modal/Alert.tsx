import { Show } from "solid-js";

import Modal from "./Modal";

interface AlertProps {
    open: boolean;
    title: string;
    message: string;
    closeLabel?: string;
    onClose?: () => void;
}

export default function Alert(
    props: AlertProps,
) {
    return (
        <Modal
            open={props.open}
            title={props.title}
            onClose={props.onClose}
        >
            <div class="padding">
                <p style={{ "white-space": "pre-line" }}>
                    {props.message}
                </p>
            </div>

            <nav class="footer">
                <button
                    type="button"
                    onClick={() =>
                        props.onClose?.()
                    }
                >
                    {props.closeLabel ?? "OK"}
                </button>
            </nav>
        </Modal>
    );
}
