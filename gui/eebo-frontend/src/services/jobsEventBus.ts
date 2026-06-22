import { pushToast } from "../state/toast.store";
import { API } from "./jobsApi";

class JobsEventBus {
  private source: EventSource | null = null;

  connect() {
    if (this.source) return;

    console.log("[JobsEventBus] pre event stream create")
    this.source = new EventSource(
      API.base + API.jobs.stream
    );
    console.log("[JobsEventBus] created event stream")

    this.source.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("[JobsEventBus.onmessage]", data)
      pushToast({
        type: "info",
        message: data.message ?? event.data,
      });
    };

    this.source.onerror = (e) => {
      console.error("[JobsEventBus.onerror]", e)
      this.disconnect();
      pushToast({
        type: "error",
        message: "Event stream disconnected",
      });
    };
  }

  disconnect() {
    console.log('[JobsEventBus] disconnect')
    this.source?.close();
    this.source = null;
  }
}

export const jobsEventBus = new JobsEventBus();
// Then client eventBus.connect();
