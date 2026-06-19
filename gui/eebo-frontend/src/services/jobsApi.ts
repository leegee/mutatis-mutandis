// src/lib/api/client.ts

import { loadJson } from "../lib/loadJson";
import { pushToast } from "../state/toast.store";

export const API = {
  base: "http://localhost:8000",

  jobs: {
    enqueue: "/jobs/enqueue",
    status: "/jobs/status",
    list: "/jobs/list",
    cancel: "/jobs/cancel",
  },

  concepts: {
    list: "/concepts",
    create: "/concepts/create_and_run",
    runTier2: "/concepts/run/tier2",
    runTier3: "/concepts/run/tier3",
  },

  system: {
    health: "/health",
  }
} as const;

export async function getJson<T>(path: string, label?: string): Promise<T> {
  try {
    return await loadJson<T>(`${ API.base }${ path }`, label);
  } catch (e: any) {
    pushToast({
      type: "error",
      message: e?.message ?? String(e),
    });
    throw e;
  }
}

export async function postJson<T>(
  path: string,
  body: unknown,
  label?: string
): Promise<T> {
  try {
    const res = await fetch(`${ API.base }${ path }`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const text = await res.text();

    if (!res.ok) {
      throw new Error(`${ label ?? path }: HTTP ${ res.status } ${ text }`);
    }

    const json = JSON.parse(text);

    pushToast({
      type: "success",
      message: label ?? "Operation complete",
    });

    return json as T;
  } catch (e: any) {
    pushToast({
      type: "error",
      message: e?.message ?? String(e),
    });
    throw e;
  }
}
