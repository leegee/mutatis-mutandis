import { For } from "solid-js";

import type { DefaultJob } from "./BaseJob.type";

interface DefaultJobsTableProps {
  jobs: DefaultJob[];
}

export default function DefaultJobsTable(props: DefaultJobsTableProps) {
  return (
    <>
      <h3>Standard Jobs</h3>

      <table class="table">
        <thead>
          <tr>
            <th>Concept</th>
            <th>Type</th>
            <th>Status</th>
            <th>Stage</th>
            <th>Attempts</th>
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
                <td>{job.job_type}</td>
                <td>{job.status}</td>
                <td>{job.stage}</td>
                <td>{job.attempts}</td>
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
