// src/lib/toast/store.ts

import { createStore } from "solid-js/store";

export type Toast = {
  id: number;
  type: "success" | "error" | "info";
  message: string;
};

const [toasts, setToasts] = createStore<Toast[]>([]);
let id = 0;

export function pushToast(t: Omit<Toast, "id">) {
  const toast: Toast = { id: ++id, ...t };

  setToasts((prev) => [...prev, toast]);

  setTimeout(() => {
    setToasts((prev) => prev.filter((x) => x.id !== toast.id));
  }, 4000);
}

export { toasts };