// @refresh reload
import { mount, StartClient } from "@solidjs/start/client";

document.body.classList.add("dark");

mount(() => <StartClient />, document.getElementById("app")!);
