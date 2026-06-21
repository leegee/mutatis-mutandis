// src/components/Jobs/JobsPanel.tsx

import { Show } from "solid-js";
import { type Resource } from "solid-js";

interface JobPanelProps {
  jobsList: Resource<any[]>;
}

export function JobPanel(props: JobPanelProps) {
  return (
    <section class="no-padding no-margin left-margin right-margin"
      style="overflow-y: auto !important; height: 100vh!important; padding-bottom: 50vh !important;">
      <progress class="small no-padding no-margin no-space"
        style={{ opacity: props.jobsList.loading ? 1 : 0 }} />
      <table class="table padding no-margin">
        <thead>
          <tr>
            <th scope="col">Concept</th>
            <th scope="col">Type</th>
            <th scope="col">Status</th>
            <th scope="col">Stage</th>
            <th scope="col">Attempts</th>
            <th scope="col">Created at</th>
            <th scope="col">Started at</th>
            <th scope="col">Finished at</th>
            <th scope="col">Last heartbeat</th>
          </tr>
        </thead>
        <tbody>
          <Show when={props.jobsList()}>
            {(jobList) =>
              jobList().map((job) => (
                <>
                  <tr>
                    <th scope="row">{job.concept}</th>
                    <td>{job.type}</td>
                    <td class={job.error ? 'error-container' : ''}>{job.status}</td>
                    <td>{job.stage}</td>
                    <td>{job.attempts}</td>
                    <td>{job.created_at}</td>
                    <td>{job.started_at}</td>
                    <td>{job.finished_at}</td>
                    <td>{job.last_heartbeat}</td>
                  </tr>
                  <Show when={job.error}>
                    <tr>
                      <td colspan="9">
                        <pre class="error-container">{job.error}</pre>
                      </td>
                    </tr>
                  </Show>
                </>
              ))
            }
          </Show>
        </tbody>
      </table>
    </section>
  );
}