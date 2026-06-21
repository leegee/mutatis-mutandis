// src/components/Jobs/index.tsx

import { createResource, onMount, onCleanup } from "solid-js";
import { ConceptCreate } from "./ConceptCreate";
import { Tabs } from "../TabsLayout";
import { JobPanel } from "./JobsPanel";
import { API, getJson } from "../../services/jobsApi";

export const POLLING_INTERVAL_MS = 10_000;

export function JobsApiComponent() {
  const [jobsList, { refetch }] = createResource(async () =>
    getJson<any[]>(API.jobs.list, "Job list")
  );

  onMount(() => {
    const timer = setInterval(() => refetch(), POLLING_INTERVAL_MS);
    onCleanup(() => clearInterval(timer));
  });

  return (
    <Tabs tabs={[
      {
        key: "jobs",
        label: "Jobs",
        icon: "sync",
        component: () => <JobPanel jobsList={jobsList} />
      },
      {
        key: "concepts",
        label: "Concepts",
        icon: "psychology",
        component: ConceptCreate
      },
    ]} />
  );
}