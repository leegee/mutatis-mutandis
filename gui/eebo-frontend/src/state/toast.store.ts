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

const DURATION_MS = 10_000;

let _id = 0;

const [toasts, setToasts] = createSignal<Toast[]>([]);

const removeToast = (id: number) => setToasts(
  toasts().filter((t) => t.id !== id)
);

export function pushToast(toast: Omit<Toast, "id" | "timeFormatted" | "durationMs">, durationMs = DURATION_MS) {
  const id = ++_id;
  setToasts((prev) => [...prev, {
    ...toast,
    id,
    timeFormatted: (new Date()).toISOString().split('T')[1].slice(0, 8),
    durationMs: (durationMs),
  }]);
}

export { toasts, removeToast };
