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
