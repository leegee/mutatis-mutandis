// src/App.tsx

import {
  createMemo,
  createResource,
  Show
} from "solid-js";

import { loadEvents } from "./services/loadEvents";

import type { SemanticEvent } from "./types/events";

import EventScatter from "./components/EventScatter";
import EventStream from "./components/EventStream";
import EventInspector from "./components/EventInspector";

import {
  selectedConcept,
  selectedSlice,
  selectedEventId
} from "./state/selection";

export default function App() {

  const [events] =
    createResource(loadEvents);

  const concepts = createMemo(() => {
    const e = events();
    if (!e) return [];

    return [...new Set(
      e.map(x => x.concept)
    )].sort();
  });


  const slicesByConcept = createMemo(() => {
    const e = events();
    if (!e) return {};

    const out: Record<string, string[]> = {};

    for (const ev of e) {
      if (!out[ev.concept]) {
        out[ev.concept] = [];
      }

      if (!out[ev.concept].includes(ev.slice)) {
        out[ev.concept].push(ev.slice);
      }
    }

    return out;
  });

  const filteredVisibleEvents = createMemo(() => {
    const e = events();

    if (!e) return [];

    return e.filter(ev => {
      const concept =
        selectedConcept();

      const slice =
        selectedSlice();

      if (concept && ev.concept !== concept) {
        return false;
      }

      if (slice && ev.slice !== slice) {
        return false;
      }

      return true;
    });
  });

  // O(1) event lookup
  const eventIndex = createMemo(() => {
    const e = events();
    if (!e) return {};

    const out: Record<string, SemanticEvent> = {};

    for (const ev of e) {
      out[ev.id] = ev;
    }

    return out;
  });

  const selectedEvent = createMemo(() => {
    const id = selectedEventId();
    if (!id) return null;

    return eventIndex()[id] ?? null;
  });

  return (
    <>
      <nav class="left">
        <EventStream
          concepts={concepts()}
          slicesByConcept={slicesByConcept()}
        />
      </nav>

      <main class="responsive max">
        <Show when={events()}>
          <div class="grid" style="max-heigh: 100%; overflow:none">
            <div class="s9 border">
              <EventScatter events={filteredVisibleEvents()} />
            </div>

            <div class="s3">
              <EventInspector event={selectedEvent()} />
            </div>
          </div>
        </Show>

      </main>
    </>
  );
}
