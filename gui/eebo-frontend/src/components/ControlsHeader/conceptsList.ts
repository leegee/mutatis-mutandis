import { createResource } from "solid-js";
import { listConcepts } from "../../services/db";
import { dbReady } from "../../state/tier2data.store";

// TODO: useConceptsList

export const conceptsList = () => conceptsResource() ?? [];

export const [conceptsResource] = createResource(
    dbReady,
    async (ready) => {
        if (!ready)
            return [];

        return listConcepts();
    }
);