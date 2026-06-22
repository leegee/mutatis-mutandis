// src/services/jobsApi.ts

import { loadJson } from "../lib/loadJson";
import { pushToast } from "../state/toast.store";

export const API = {
  base: "http://localhost:8000",

  jobs: {
    enqueue: "/jobs/enqueue",
    status: "/jobs/status",
    list: "/jobs/",
    cancel: "/jobs/cancel",
    events: "/jobs/:job_id/events"
  },

  concepts: {
    list: "/concepts",
    create_and_run: "/concepts/create_and_run",
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
      throw new Error(`${ label ?? path }:\n HTTP ${ res.status } - ${ res.statusText } - ${ text }`);
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


export async function jobEventStreat(jobId: string) {
  const source = new EventSource(`/jobs/${ jobId }/events`);

  source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);
    pushToast({ type: 'info', message: event.data })
  };

  source.onerror = (e) => {
    source.close();
    console.error(e);
    pushToast({ type: 'error', message: 'Event bus error' })
  };
}
