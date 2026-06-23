// src/lib/state/toast.store.ts

import { createSignal } from "solid-js";

export type ToastType = "info" | "error" | "success";

export interface Toast {
  id: number;
  timeFormatted: string;
  type: ToastType;
  message: string;
  durationMs: number;
}

let _id = 0;

const [toasts, setToasts] = createSignal<Toast[]>([]);

const removeToast = (id: number) => setToasts(
  toasts().filter((t) => t.id !== id)
);

export function pushToast(toast: Omit<Toast, "id" | "durationMs" | "timeFormatted">, durationMs = 4_000) {
  const id = ++_id;
  const timeFormatted = (new Date()).toISOString().split('T')[1].slice(0, 8);

  setToasts((prev) => [...prev, { ...toast, id, timeFormatted, durationMs }]);
}

export { toasts, removeToast };
