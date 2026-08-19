import { showAlert, showConfirm, showCustom, showPrompt } from "./modal";

export { default as Modal } from "./BaseModal";

export function useAlert() {
    return showAlert;
}

export function useConfirm() {
    return showConfirm;
}

export function usePrompt() {
    return showPrompt;
}

export function useModal() {
    return showCustom;
}
