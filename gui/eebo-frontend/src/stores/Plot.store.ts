import { createSignal } from "solid-js";

export const [selected, setSelected] = createSignal({
    token: null,
    year: null,
    color: "#888"
});

export const [animate, setAnimate] = createSignal(false);
