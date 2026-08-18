import {
    createSignal,
    type JSX,
} from "solid-js";

export type ModalRequest =
    | {
        kind: "alert";
        title?: string;
        message: string;
        resolve: () => void;
    }
    | {
        kind: "confirm";
        title?: string;
        message: string;
        resolve: (value: boolean) => void;
    }
    | {
        kind: "prompt";
        title?: string;
        message: string;
        defaultValue?: string;
        resolve: (value: string | null) => void;
        // }
        // | {
        //     kind: "custom";
        //     content: JSX.Element;
        //     resolve: () => void;
    };

const [current, setCurrent] =
    createSignal<ModalRequest>();

export function modalState() {
    return current;
}

export function showAlert(
    message: string,
    title?: string,
): Promise<void> {
    return new Promise((resolve) => {
        setCurrent({
            kind: "alert",
            title,
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
): Promise<boolean> {
    return new Promise((resolve) => {
        setCurrent({
            kind: "confirm",
            title,
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
): Promise<string | null> {
    return new Promise((resolve) => {
        setCurrent({
            kind: "prompt",
            title,
            message,
            defaultValue,
            resolve: (value) => {
                setCurrent(undefined);
                resolve(value);
            },
        });
    });
}
