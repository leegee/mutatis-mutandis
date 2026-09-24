import { MetaProvider, Title } from "@solidjs/meta";
import { Router, useLocation } from "@solidjs/router";
import { FileRoutes } from "@solidjs/start/router";
import { createSignal, Suspense } from "solid-js";

import ControlsHeader from './components/ControlsHeader/';
import ModalHost from "./components/Modal/ModalHost";

import "beercss/dist/cdn/beer.min.css";
import "./app.css";
import "./nav-menu.css";

function Navigation() {
	const location = useLocation();
	const [menuOpen, setMenuOpen] = createSignal(false);
	const toggleMenu = () => setMenuOpen((open) => !open);
	const closeMenu = () => setMenuOpen(false);
	const isActive = (path: string) => location.pathname === path;

	return (
		<div class="navigation-button-menu surface-container-lowest round">
			<button type="button" class="margin transparent" onClick={toggleMenu}>
				<i>{menuOpen() ? "menu_open" : "menu"}</i>
				<span>Navigation</span>
				<i>{menuOpen() ? "arrow_drop_up" : "arrow_drop_down"}</i>
			</button>

			{menuOpen() && (
				// biome-ignore lint/a11y/useKeyWithClickEvents: <no need>
				<menu class="margin" onClick={closeMenu}>
					<li classList={{ active: isActive("/") }}>
						<a href="/">
							<i>home</i>
							<span>Home</span>
						</a>
					</li>

					<li classList={{ active: isActive("/") }}>
						<a href="/browser">
							<i>network_node</i>
							<span>Browser</span>
						</a>
					</li>

					<li classList={{ active: isActive("/") }}>
						<a href="/scatter">
							<i>network_node</i>
							<span>Scatter</span>
						</a>
					</li>

					<li classList={{ active: isActive("/") }}>
						<a href="/linear">
							<i>network_node</i>
							<span>Linear</span>
						</a>
					</li>
				</menu>
			)}
		</div>
	);
}

export default function App() {
	return (
		<>
			<Router
				root={(props) => (
					<MetaProvider>
						<Title>Visualisation</Title>

						<main class="responsive max no-padding background">
							<Suspense>
								<ControlsHeader />
								<Navigation />
								{props.children}
							</Suspense>
						</main>
					</MetaProvider>
				)}
			>
				<FileRoutes />
			</Router>

			<ModalHost />
		</>
	);
}
