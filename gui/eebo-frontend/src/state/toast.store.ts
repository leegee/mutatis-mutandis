// src/lib/state/toast.store.ts

import { createSignal } from "solid-js";

export type ToastType = "info" | "error" | "success";

export interface Toast {
  id: number;
  timeFormatted: string;
  type: ToastType;
  message: string;
}

let _id = 0;

const [toasts, setToasts] = createSignal<Toast[]>([]);

const removeToast = (id: number) => setToasts(
  toasts().filter((t) => t.id !== id)
);

export function pushToast(toast: Omit<Toast, "id">, durationMs = 8_000) {
  const id = ++_id;

  toast.timeFormatted = (new Date()).toISOString().split('T')[1].slice(0, 8);

  setToasts((prev) => [...prev, { ...toast, id }]);

  setTimeout(() => {
    removeToast(id);
  }, durationMs);
}

export { toasts, removeToast };
