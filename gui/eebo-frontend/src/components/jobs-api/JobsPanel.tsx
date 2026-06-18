// src/components/jobs-api/JobPanel.tsx

import { createSignal, onMount } from "solid-js";
import { API, getJson } from "../../services/jobsApi";
import type { Tab } from "../TabsLayout";

interface Props {
  tab: Tab;
}

export function JobPanel(props: Props) {
  const [jobs, setJobs] = createSignal<any[]>([]);

  const refresh = async () => {
    const data = await getJson<any[]>(API.jobs.list, "Job list");
    setJobs(data);
  };

  onMount(() => {
    refresh();
    setInterval(refresh, 3000);
  });

  return (
    <article class="padding">
      <header>
        <h2>Jobs</h2>
      </header>

      {jobs().map(job => (
        <div class="job row padding">
          <div class='s4'>{job.type}</div>
          <div class='s4'>{job.status}</div>
        </div>
      ))}
    </article>
  );
}
