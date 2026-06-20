// src/components/Jobs/index.tsx

import { ConceptCreate } from "./ConceptCreate";
import { Tabs } from "../TabsLayout";
import { JobPanel } from "./JobsPanel";

export function JobsApiComponent() {
  return (
    <Tabs tabs={[
      {
        key: "concepts",
        label: "Concepts",
        icon: "psychology",
        component: ConceptCreate
      },
      {
        key: "jobs",
        label: "Jobs",
        icon: "sync",
        component: JobPanel
      }
    ]} />
  );
}
