import { createSignal } from "solid-js";
import { A, useLocation } from "@solidjs/router";
import { Icon } from "./Icon";

const navItems = [
    { path: "/scatter", icon: "scatter_plot", label: "Scatter Plot" },
    { path: "/aggregates", icon: "crowdsource", label: "Aggregates" },
    { path: "/clusters", icon: "view_cozy", label: "Cluster Report" },
    { path: "/graph2", icon: "orbit", label: "FDG" },
    { path: "/table", icon: "view_column", label: "Neighbourhood Table" },
    { path: "/diachronic", icon: "calendar_view_week", label: "Diachronic Chart" },
] as const;

export default function AppNav() {
    const location = useLocation();
    const [open, setOpen] = createSignal(false);
    const isActive = (path: string) => location.pathname === path;

    return (
        <nav id="app-nav" class={`surface-container-low left no-margin top-padding scroll small-elevate ${ open() ? "full" : "small" }`} >
            <header class="center-align top-margin tiny-margin no-padding">
                <button
                    class="extra transparent no-padding no-margin"
                    onClick={() => setOpen(!open())}
                >
                    <Icon style="scale:3; margin-top: -.2rem; opacity: 0.5" />
                </button>
            </header>

            {navItems.map((item) => (
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

            {/* <a
                onClick={() => setOpenHelp(!openHelp())}
                class="extra-padding bottom-padding"
            >
                <i>help</i>
                <span>Guide</span>
            </a> */}
        </nav>
    );
}