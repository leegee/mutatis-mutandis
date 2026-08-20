import { MetaProvider, Title } from "@solidjs/meta";
import { Router, useLocation } from "@solidjs/router";
import { FileRoutes } from "@solidjs/start/router";
import { createSignal, Suspense } from "solid-js";

import ModalHost from "./components/Modal/ModalHost";
import ProjectExport from "./components/ProjectExport";
import ProjectImport from "./components/ProjectImport";
import RelationshipImportButton from "./components/RelationshipImport";

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
		<div class="navigation-button-menu">
			<button type="button" class="surface-container-lowest margin" onClick={toggleMenu}>
				<i>{menuOpen() ? "menu_open" : "menu"}</i>
				<span>Navigation</span>
				<i>{menuOpen() ? "arrow_drop_up" : "arrow_drop_down"}</i>
			</button>

			{menuOpen() && (
				<menu class="margin" onClick={closeMenu}>
					<li classList={{ active: isActive("/") }}>
						<a href="/">
							<i>network_node</i>
							<span>Map</span>
						</a>
					</li>

					<li classList={{ active: isActive("/entities") }}>
						<a href="/entities">
							<i>circle</i>
							<span>Entities</span>
						</a>
					</li>

					<li classList={{ active: isActive("/relations") }}>
						<a href="/relations">
							<i>arrow_and_edge</i>
							<span>Relations</span>
						</a>
					</li>

					<li>
						<i>content_paste</i>
						<RelationshipImportButton />
					</li>

					{/* <li classList={{ active: isActive("/project") }}>
						<a href="/project" onClick={closeMenu}>
							<i>folder_open</i>
							<span>Import/Export</span>
						</a>
					</li> */}

					<li>
						<i>file_open</i>
						<ProjectImport />
					</li>

					<li>
						<i>download</i>
						<ProjectExport />
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
