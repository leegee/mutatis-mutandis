import { createSignal } from "solid-js";
import {
  exportSelectedEvents,
  downloadJson,
  downloadCsv,
  type ExportData,
  copyTextToClipboard,
  copyToClipboard,
  getEnrichedSelectedEvents,
  type EnrichedEvent,
} from "../lib/eventExport";
import { controls } from "../state/controls.store";
import { cluster2groq } from "../services/groqApi";
import { labelsActions } from "../state/labels.actions";
import type { PointData } from "./ScatterPlot/types";

export default function ExportSelectedEvents() {
  const [isExporting, setIsExporting] = createSignal(false);
  const [exportedData, setExportedData] = createSignal<ExportData | null>(null);

  const handleExport = async () => {
    if (!exportedData()) {
      setIsExporting(true);
      try {
        const data = await exportSelectedEvents();
        // console.log(data);
        setExportedData(data);
      } finally {
        setIsExporting(false);
      }
    }
  };
  const allText = () => exportedData()!.events.map((e) => e.windowText || "").filter(Boolean).join("\n\n");

  const noSelection = !controls.selectedEventIds || controls.selectedEventIds.size < 1;

  return (
    <div>
      <button class="small border circle secondary-text secondary-border"
        aria-disabled={isExporting() || noSelection}
        onClick={handleExport}
      >
        <i>{isExporting() ? "hourglass_empty" : "file_export"}</i>
      </button>

      <menu class="left no-wrap">
        <li>
          <i>content_copy</i>
          <span>Copy</span>

          <menu class="left no-wrap">
            <li onClick={() => navigator.clipboard.writeText(JSON.stringify(exportedData(), null, 2))}>
              <i>data_object</i>
              <span> JSON </span>
            </li>

            <li onClick={() => copyTextToClipboard(exportedData()!)}>
              <i>table</i>
              <span> CSV </span>
            </li>

            <li onClick={() => {
              copyToClipboard(allText(), "Window texts");
            }}>
              <i>text_fields</i>
              <span>Text</span>
            </li>
          </menu>
        </li>

        <li>
          <i>download</i>
          <span>Download</span>

          <menu id="export-menu" class="left no-wrap">
            <li onClick={() => downloadJson(exportedData()!, "selected-events.json")}>
              <i>data_object</i>
              <span> JSON </span>
            </li>

            <li onClick={() => downloadCsv(exportedData()!.events, "selected-events.csv")}>
              <i>table</i>
              <span> CSV </span>
            </li>

            <li class="disabled">
              <i>database</i>
              <span> SQLite </span>
            </li>
          </menu>
        </li>

        {/* <TopicAnalysisButton
          exportedData={exportedData}
          concept={controls.conceptSelection[0]}
        /> */}

        <li onClick={async () => {
          // const selectedPoints = controls.selectedPoints
          //   .filter(
          //     (selectedPoint): selectedPoint is EnrichedEvent & { nx: number; ny: number } =>
          //       typeof selectedPoint.nx === "number" && typeof selectedPoint.ny === "number"
          //   )
          //   .map((p) => ({
          //     id: p.event_id,
          //     x: p.nx,
          //     y: p.ny,
          //   }));
          // console.log('[ExportSelectedEvents] selected points:', selectedPoints)

          console.log('[ExportSelectedEvents] Call cluster2groq')
          const result = await cluster2groq({
            concept: exportedData()!.events[0].concept.toLowerCase(),
            rawText: allText()
          });

          const labelText = result.sense_name;

          const success = labelsActions.createFromCluster(
            controls.selectedPoints,
            labelText
          );

          if (!success) {
            alert("Label too close to existing cluster");
            return;
          }

          alert(`${ result.sense_name }\n\n${ result.description }`);
        }}>
          <i>new_label</i>
          <span>Label with Groq</span>
        </li>
      </menu>
    </div >
  );
}