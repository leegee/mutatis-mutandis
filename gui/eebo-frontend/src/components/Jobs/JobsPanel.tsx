// JobsPanel.tsx

import { Show, type Resource } from "solid-js";

import { jobTypeGuards, type Job } from "./BaseJob.type";
import TopicJobsTable from "./TopicJobsTable";
import DefaultJobsTable from "./DefaultJobsTable";

interface JobsPanelProps {
  jobsList: Resource<Job[]>;
}

export function JobsPanel(props: JobsPanelProps) {
  const jobs = () => props.jobsList();

  const topicJobs = () => (props.jobsList() ?? []).filter(jobTypeGuards.topic_analysis)
  const defaultJobs = () => jobs()?.filter(j => j.job_type !== "topic_analysis") ?? [];

  return (
    <section class="no-padding no-margin left-margin right-margin"
      style="overflow-y: auto; height: 100vh; padding-bottom: 50vh;">

      <Show when={!props.jobsList.loading}>
        <>
          <TopicJobsTable jobs={topicJobs()} />
          <DefaultJobsTable jobs={defaultJobs()} />
        </>
      </Show>

    </section>
  );
}
