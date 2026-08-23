
import { resetProject } from "~/db/respository";
import { useConfirm } from "../Modal";

export default function ProjectReset() {
    const confirm = useConfirm();
    async function reset() {
        const ok = await confirm("Do you wish to create a blank project? This will wipe the current project.");
        if (ok) resetProject();
    }

    return (
        <button type="button" class="small transparent no-padding" onClick={reset}>
            Reset project
        </button>
    );
}
