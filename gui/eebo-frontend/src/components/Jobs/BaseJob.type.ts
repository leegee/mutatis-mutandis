export type BaseJob = {
  job_id: string;
  concept: string;
  job_type: string;
  status: string;
  stage?: string;
  attempts?: number;

  created_at?: string;
  started_at?: string;
  finished_at?: string;
  last_heartbeat?: string;

  error?: string;
};

export type TopicAnalysisJob = BaseJob & {
  job_type: "topic_analysis";
  document_count?: number;
  topic_count?: number;
};

export type DefaultJob = BaseJob & {
  job_type: Exclude<string, "topic_analysis">;
};

export type Job = TopicAnalysisJob | DefaultJob;


export const jobTypeGuards = {
  topic_analysis: (job: Job): job is TopicAnalysisJob =>
    job.job_type === "topic_analysis",
};
