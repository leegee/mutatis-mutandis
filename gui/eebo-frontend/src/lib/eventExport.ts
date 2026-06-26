import type { SemanticEvent } from '../types';
import { controls } from "../state/controls.store";
import { queryEventById } from "../services/db";
import { fetchWindowBatch } from "../services/tokenWindowBatchApi";
import { setWindowCache, getWindow, } from "../services/windowCache";

export type EnrichedEvent = SemanticEvent & {
  event_id: string;
  token_idx: number;
  pub_year: number;
  windowText?: string;
  windowKey: string;
  gnx?: number;
  gny?: number;
  nx?: number;
  ny?: number;
};

export type ExportData = {
  selectedCount: number;
  exportedAt: string;
  events: EnrichedEvent[];
  groupedByDoc: Record<string, EnrichedEvent[]>;
};

// Adds window text to an event using the cache (or fetches if missing)
export async function enrichEventWithWindow(e: any): Promise<EnrichedEvent> {
  const key = `${ e.doc_id }:${ Number(e.token_idx) }`;

  let windowText = getWindow(key);

  if (!windowText) {
    // Optional: fallback fetch single window if not in cache
    try {
      const batch = await fetchWindowBatch([{ docId: e.doc_id, tokenIdx: Number(e.token_idx) }]);
      if (batch.results?.[0]) {
        windowText = batch.results[0].content;
        setWindowCache(key, windowText);
      }
    } catch (err) {
      console.warn("Failed to fetch window for", key, err);
    }
  }

  return {
    id: e.id || e.event_id,
    event_id: e.event_id || e.id,
    doc_id: e.doc_id,
    token_idx: Number(e.token_idx),
    token: e.token,
    pub_year: e.pub_year,
    windowText: windowText || undefined,
    windowKey: key,
    ...e, // keep any extra fields
  };
}

// Gets all currently selected events enriched with window text
export async function getEnrichedSelectedEvents(): Promise<EnrichedEvent[]> {
  const selectedIds = controls.selectedEventIds
    ? Array.from(controls.selectedEventIds)
    : [];

  if (!selectedIds.length) return [];

  const events = await Promise.all(
    selectedIds.map((id) => queryEventById(id))
  );

  console.log("[getEnrichedSelectedEvents]", events)

  const cleanEvents = events.filter(Boolean);

  return Promise.all(cleanEvents.map(enrichEventWithWindow));
}


// Main export function — returns structured data ready for JSON, CSV, etc.
export async function exportSelectedEvents(): Promise<ExportData> {
  const enriched = await getEnrichedSelectedEvents();

  const groupedByDoc: Record<string, EnrichedEvent[]> = {};

  for (const ev of enriched) {
    if (!groupedByDoc[ev.doc_id]) groupedByDoc[ev.doc_id] = [];
    groupedByDoc[ev.doc_id].push(ev);
  }

  // Sort events within each document
  Object.keys(groupedByDoc).forEach((doc) => {
    groupedByDoc[doc].sort((a, b) => a.token_idx - b.token_idx);
  });

  return {
    selectedCount: enriched.length,
    exportedAt: new Date().toISOString(),
    events: enriched,
    groupedByDoc,
  };
}

export function downloadJson(data: ExportData, filename = "selected-events.json") {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function downloadCsv(events: EnrichedEvent[], filename = "selected-events.csv") {
  if (!events.length) return;

  const headers = ["doc_id", "token_idx", "token", "pub_year", "windowText"];
  const rows = events.map((e) => [
    e.doc_id,
    e.token_idx,
    `"${ (e.token || "").replace(/"/g, '""') }"`,
    e.pub_year,
    `"${ (e.windowText || "").replace(/"/g, '""') }"`,
  ]);

  const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");

  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}


export function copyTextToClipboard(data: ExportData) {
  if (!data) return;
  const rows = data.events;
  const headers = Object.keys(rows[0] ?? {}).join(",");
  const body = rows.map((r) => Object.values(r).map((v) => JSON.stringify(v ?? "")).join(",")).join("\n");
  copyToClipboard(`${ headers }\n${ body }`, "CSV");
}

export async function copyToClipboard(text: string, label: string) {
  try {
    await navigator.clipboard.writeText(text);
    console.log(`${ label } copied to clipboard`);
  } catch (err) {
    console.error("Failed to copy", err);
  }
};
