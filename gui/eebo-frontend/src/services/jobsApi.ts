// src/services/jobsApi.ts

export const API = {
  base: "http://localhost:8000",

  jobs: {
    enqueue: "/jobs/enqueue",
    status: "/jobs/status",
    list: "/jobs/",
    cancel: "/jobs/cancel",
    events: "/jobs/:job_id/events",
    stream: "/stream/"
  },

  concepts: {
    list: "/concepts",
    create_and_run: "/concepts/create_and_run",
    runTier2: "/concepts/run/tier2",
    runTier3: "/concepts/run/tier3",
  },

  system: {
    health: "/health",
  },

  topic: {
    analyse: "/topic"
  }
} as const;

