import { MetaProvider, Title } from "@solidjs/meta";
import { Router, useLocation } from "@solidjs/router";
import { FileRoutes } from "@solidjs/start/router";
import { createSignal, Suspense } from "solid-js";
import "beercss/dist/cdn/beer.min.css";

import "./app.css";
import "./nav-menu.css";
import ModalHost from "./components/Modal/ModalHost";

function Navigation() {
	const location = useLocation();
	const [menuOpen, setMenuOpen] = createSignal(false);
	const toggleMenu = () => setMenuOpen((open) => !open);
	const closeMenu = () => setMenuOpen(false);
	const isActive = (path: string) => location.pathname === path;

	return (
		<div class="navigation-button-menu">
			<button type="button" class="surface-container-lowest margin" onClick={toggleMenu}>
				<i>{menuOpen() ? "menu_open" : "menu"}</i>
				<span>Navigation</span>
				<i>{menuOpen() ? "arrow_drop_up" : "arrow_drop_down"}</i>
			</button>

			{menuOpen() && (
				<menu class="margin">
					<li classList={{ active: isActive("/") }}>
						<a href="/" onClick={closeMenu}>
							<i>network_node</i>
							<span>Map</span>
						</a>
					</li>

					<li classList={{ active: isActive("/entities") }}>
						<a href="/entities" onClick={closeMenu}>
							<i>circle</i>
							<span>Entities</span>
						</a>
					</li>

					<li classList={{ active: isActive("/relations") }}>
						<a href="/relations" onClick={closeMenu}>
							<i>arrow_and_edge</i>
							<span>Relations</span>
						</a>
					</li>

					<li classList={{ active: isActive("/project") }}>
						<a href="/project" onClick={closeMenu}>
							<i>folder_open</i>
							<span>Import/Export</span>
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
						<Title>Research</Title>
						{/* <SideNavigation /> */}

						<main class="responsive max no-padding background">
							<Suspense>
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
