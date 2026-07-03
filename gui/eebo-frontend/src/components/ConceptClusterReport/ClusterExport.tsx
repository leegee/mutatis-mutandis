import { createSignal, Show } from "solid-js";
import type { ClusterReport } from "./loadClusterReport";

interface Props {
    clusters: ClusterReport;
}

export default function ClusterExport(props: Props) {
    const [exporting, setExporting] = createSignal(false);

    const buildExportPayload = () => ({
        ...props.clusters,
        exported_at: new Date().toISOString(),
    });

    const handleCopyJson = async () => {
        setExporting(true);
        try {
            const text = JSON.stringify(buildExportPayload(), null, 2);
            await navigator.clipboard.writeText(text);
        } finally {
            setExporting(false);
        }
    };

    const handleDownloadJson = async () => {
        setExporting(true);
        try {
            const text = JSON.stringify(buildExportPayload(), null, 2);
            const blob = new Blob([text], { type: "application/json" });
            const url = URL.createObjectURL(blob);

            const a = document.createElement("a");
            a.href = url;
            a.download = `${ props.clusters.concept }_clusters.json`;

            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);

            URL.revokeObjectURL(url);
        } finally {
            setExporting(false);
        }
    };

    return (
        <>
            <button onClick={handleCopyJson} disabled={exporting()} class="border small">
                <i>content_copy</i>
                <span>Copy JSON</span>
            </button>

            <button onClick={handleDownloadJson} disabled={exporting()} class="border small">
                <i>download</i>
                <span> Download JSON </span>
            </button>

            <Show when={exporting()}>
                <progress />
            </Show>
        </>
    );
}
