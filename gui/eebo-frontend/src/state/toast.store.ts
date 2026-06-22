// src/lib/state/toast.store.ts

import { createSignal } from "solid-js";

export type ToastType = "info" | "error" | "success";

export interface Toast {
  id: number;
  type: ToastType;
  message: string;
}

let _id = 0;

const [toasts, setToasts] = createSignal<Toast[]>([]);

export { toasts };

export function pushToast(toast: Omit<Toast, "id">, durationMs = 8_000) {
  const id = ++_id;
  setToasts((prev) => [...prev, { ...toast, id }]);
  setTimeout(() => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, durationMs);
}
