import type { PickingInfo } from "@deck.gl/core";
import type { SelectionController } from "./SelectionController";
import type { Id } from "./types";

export class DeckClickPlugin<T extends { event_id: Id }> {
    constructor(
        deck: any,
        private controller: SelectionController<T>
    ) {
        deck.setProps({
            onClick: this.onClick,
        });
    }

    private onClick = (info: PickingInfo<T>) => {
        if (!info.object) return;

        // console.debug("[deck-click] background-click", info);

        this.controller.dispatch({
            type: "click",
            payload: info.object,
        });
    };
}
