import { createStore } from "solid-js/store";

const [windowCache, setWindowCacheStore] = createStore<Record<string, string>>({});

export function getWindowCacheStore() {
  return windowCache;
}

export function setWindowCache(key: string, value: string) {
  setWindowCacheStore(key, value);
}

export function hasWindowCache(key: string) {
  return key in windowCache;
}

export function clearWindowCache() {
  setWindowCacheStore({});
}

export function getWindow(eventId: string) {
  return windowCache[eventId] ?? null;
}
