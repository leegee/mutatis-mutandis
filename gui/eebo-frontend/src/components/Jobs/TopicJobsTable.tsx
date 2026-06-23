import { For } from "solid-js";

import type { TopicAnalysisJob } from "./BaseJob.type";

interface TopicJobsTableProps {
  jobs: TopicAnalysisJob[];
}

export default function TopicJobsTable(props: TopicJobsTableProps) {
  return (
    <>
      <h3>Topic Analysis Jobs</h3>

      <table class="table">
        <thead>
          <tr>
            <th>Concept</th>
            <th>Status</th>
            <th>Documents</th>
            <th>Topics</th>
            <th>Created</th>
            <th>Started</th>
            <th>Finished</th>
          </tr>
        </thead>

        <tbody>
          <For each={props.jobs}>
            {(job) => (
              <tr>
                <td>{job.concept}</td>
                <td>{job.status}</td>
                <td>{job.document_count}</td>
                <td>{job.topic_count ?? "-"}</td>
                <td>{job.created_at}</td>
                <td>{job.started_at}</td>
                <td>{job.finished_at}</td>
              </tr>
            )}
          </For>
        </tbody>
      </table>
    </>
  );
}
