// src/components/Jobs/index.tsx

import { createResource, onMount, onCleanup } from "solid-js";
import { ConceptCreate } from "./ConceptCreate";
import { Tabs } from "../TabsLayout";
import { JobsPanel } from "./JobsPanel";
import { API } from "../../services/jobsApi";
import { jobsEventBus } from "../../services/jobsEventBus";
import { getJson } from "../../lib/json";

export const POLLING_INTERVAL_MS = 10_000;

export function JobsApiComponent() {
  const [jobsList, { refetch }] = createResource(async () =>
    getJson<any[]>(API.jobs.list, "Job list")
  );

  onMount(() => {
    jobsEventBus.connect();
    const timer = setInterval(() => refetch(), POLLING_INTERVAL_MS);
    onCleanup(() => clearInterval(timer));
  });

  return (
    <Tabs tabs={[
      {
        key: "jobs",
        label: "Jobs",
        icon: "sync",
        component: () => <JobsPanel jobsList={jobsList} />
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