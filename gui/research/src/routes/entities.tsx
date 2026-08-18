import { clientOnly } from "@solidjs/start";

export default clientOnly(
    () => import("~/components/EntitiesPage"),
);
