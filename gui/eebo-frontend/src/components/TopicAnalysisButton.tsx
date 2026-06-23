// gui/eebo-frontend/src/components/TopicAnalysisButton.tsx
import { createSignal, Show } from "solid-js";
import { postJson } from "../lib/json";
import { pushToast } from "../state/toast.store";
import type { ExportData } from "../lib/eventExport";
import { API } from "../services/jobsApi";

interface Props {
  exportedData: () => ExportData | null;
  concept?: string;
}

export default function TopicAnalysisButton(props: Props) {
  const [isRunning, setIsRunning] = createSignal(false);

  const handleTopicAnalysis = async () => {
    const data = props.exportedData();
    if (!data || !data.events?.length) {
      pushToast({ type: "error", message: "No events selected for analysis" });
      return;
    }

    const concept = props.concept || data.events[0]?.concept || "unknown";

    const documents = data.events
      .map(e => e.windowText || "")
      .filter(text => text.length > 30); // basic filter

    if (documents.length === 0) {
      pushToast({ type: "error", message: "No usable text found in selection" });
      return;
    }

    setIsRunning(true);

    const body = {
      concept: concept.toLowerCase(),
      documents: documents,
      label: `topic-${ concept }-${ new Date().toISOString().slice(0, 10) }`,
      min_topic_size: 5,
      use_sentence_chunking: true,
    };
    console.log("[topic-button] Sending", JSON.stringify(body, null, 2))

    try {
      const result = await postJson<{ job_id: string; status: string }>(
        API.topic.analyse,
        body,
        `Start topic analysis for "${ concept }"`
      );

      console.log("[topic-button] Result", result)

      pushToast({
        type: "success",
        message: `Topic analysis queued. Job ID: ${ result.job_id }`,
      });

      // window.open(`/jobs/${ result.job_id }`, "_blank");

    } catch (err) {
      console.error(err);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <li onClick={handleTopicAnalysis} classList={{ disabled: isRunning() }}>
      <i>{isRunning() ? "hourglass_empty" : "scatter_plot"}</i>
      <span>Semantic Topics</span>
      <Show when={isRunning()}>
        <span class="small secondary-text">(running...)</span>
      </Show>
    </li>
  );
}