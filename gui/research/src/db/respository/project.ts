import { ResearchProject, ProjectMetadata } from "~/domain/project";
import { validateProject, isValidProject } from "~/domain/validateProject";
import { getDatabase } from "../database";

export async function importProject(value: unknown,): Promise<void> {
  const validation = validateProject(value);

  if (!validation.valid) {
    throw new Error(
      validation.errors
        .map((error) => `${ error.path }: ${ error.message }`,)
        .join("\n"),
    );
  }

  if (!isValidProject(value)) {
    // Should never happen because validateProject()has already succeeded.
    throw new Error("Invalid research project.");
  }

  const project = value;

  // project is now ResearchProject
  const db = getDatabase();

  await db.transaction(
    "rw",
    db.entities,
    db.relations,
    db.evidence,
    db.projectMetadata,
    async () => {
      // ...
    },
  );
}

export async function exportProject(): Promise<ResearchProject> {
  const db = getDatabase();

  const [
    metadataRecord,
    entities,
    relations,
    evidence,
  ] = await Promise.all([
    db.projectMetadata.get("project"),
    db.entities.toArray(),
    db.relations.toArray(),
    db.evidence.toArray(),
  ]);

  if (!metadataRecord) {
    throw new Error("Project metadata has not been initialized.",);
  }

  const {
    id: _id,
    ...metadata
  } = metadataRecord;

  return {
    version: 1,
    metadata,
    entities,
    relations,
    evidence,
  };
}


export async function getProjectMetadata(): Promise<ProjectMetadata | undefined> {
  const record = await getDatabase()
    .projectMetadata
    .get("project");

  if (!record) {
    return undefined;
  }

  const {
    id: _id,
    ...metadata
  } = record;

  return metadata;
}

export async function saveProjectMetadata(metadata: ProjectMetadata,): Promise<void> {
  await getDatabase()
    .projectMetadata
    .put({
      id: "project",
      ...metadata,
    });
}

