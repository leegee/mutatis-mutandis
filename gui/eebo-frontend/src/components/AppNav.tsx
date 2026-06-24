import { createSignal } from "solid-js";
import { A, useLocation } from "@solidjs/router";
import { Icon } from "./Icon";
import { routes } from "../routes";
import { openHelp, setOpenHelp } from "../state/help.store";


export default function AppNav() {
    const location = useLocation();
    const [open, setOpen] = createSignal(false);
    const isActive = (path: string) => location.pathname === path;

    return (
        <nav id="app-nav" class={`surface-container-low left no-margin top-padding scroll small-elevate ${ open() ? "full" : "small" }`} >
            <header class="fixed left-align top-margin tiny-margin no-padding">
                <button
                    class="extra transparent no-padding no-margin"
                    onClick={() => setOpen(!open())}
                >
                    <Icon style="scale:3; margin-top: -.2rem; opacity: 0.5" />
                </button>
            </header>

            {routes.map((item) => (
                <A
                    href={item.path}
                    classList={{
                        active: isActive(item.path),
                        button: true,
                        transparent: true,
                        "no-border": true,
                        "no-padding": true,
                        "no-margin": true,
                        "no-space": true,
                    }}
                >
                    <i>{item.icon}</i>
                    <span>{item.label}</span>
                </A>
            ))}

            <hr class="max surface-container-low" />

            <a class="no-margin no-space no-padding no-border button transparent" onClick={() => setOpenHelp(!openHelp())} >
                <i>help</i>
                <span>Guide</span>
            </a>
        </nav>
    );
}