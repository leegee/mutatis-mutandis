import { API } from "../services/jobsApi";
import { pushToast } from "../state/toast.store";

// gui\eebo-frontend\src\lib\loadJson.ts
export async function loadJson<T = any>(url: string, label?: string): Promise<T> {
  console.log(`Loading ${ url } for ${ label ?? '-' }`);

  const res = await fetch(url);

  if (!res.ok) {
    throw new Error(`${ label ?? url }: HTTP ${ res.status } ${ res.statusText }`);
  }

  const contentType = res.headers.get("content-type");

  const text = await res.text();

  // detect Vite / HTML fallback issue
  if (!contentType?.includes("application/json")) {
    throw new Error(
      `${ label ?? url }: expected JSON from ${ url }, got ${ contentType }\n${ text.slice(0, 120) }`
    );
  }

  try {
    const json = JSON.parse(text);
    console.log(`Loaded ${ url } for ${ label ?? '-' }`);
    // console.log('JSON', json)
    return json as T;
  } catch (e) {
    throw new Error(`${ label ?? url }: invalid JSON`);
  }
}


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


