/** biome-ignore-all lint/a11y/useKeyWithClickEvents: Time is limited now */
/** biome-ignore-all lint/a11y/noStaticElementInteractions: Time is limited now */

import cytoscape, { type Core, type ElementDefinition } from "cytoscape";
import cytoscapeElk from "cytoscape-elk";
import { createEffect, createSignal, For, Match, onCleanup, onMount, Show, Switch } from "solid-js";

import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";
import { useConfirm } from "./Modal/index";

cytoscape.use(cytoscapeElk);

type ContextMenu =
	| {
		kind: "canvas";
		x: number;
		y: number;
	}
	| {
		kind: "node";
		x: number;
		y: number;
		nodeId: string;
	}
	| {
		kind: "edge";
		x: number;
		y: number;
		relationId: string;
	};

const [contextMenu, setContextMenu] = createSignal<ContextMenu>();

const [linkingFrom, setLinkingFrom] = createSignal<string>();

const typeColors: Record<string, { hue: number }> = {
	concept: { hue: 308 },
	lexeme: { hue: 120 },
	motif: { hue: 283 },
	animal: { hue: 43 },
	person: { hue: 23 },
	source: { hue: 234 },
	group: { hue: 190 },
};

function hueForType(type: string): number {
	if (typeColors[type]) return typeColors[type].hue;
	// stable fallback hue for any type not explicitly styled
	let hash = 0;
	for (let i = 0; i < type.length; i++) hash = (hash * 31 + type.charCodeAt(i)) % 360;
	return hash;
}

interface GraphViewProps {
	entities: Entity[];
	relations: Relation[];

	onSelectEntity?: (entity: Entity) => void;
	onSelectRelation?: (relation: Relation) => void;

	onAddEntity?: (position: { x: number; y: number }) => void;

	onEditEntity?: (entity: Entity) => void;
	onDeleteEntity?: (entity: Entity) => void;

	onAddRelation?: (sourceId: string, targetId: string) => void;

	onEditRelation?: (relation: Relation) => void;
	onDeleteRelation?: (relation: Relation) => void;
}


const LAYOUT_PARAMS = {
	name: "elk",
	animate: false,
	fit: true,
	padding: 40,
	nodeDimensionsIncludeLabels: true,
	elk: {
		algorithm: "layered",
		"elk.direction": "DOWN",

		// Give ELK room to work with variable node sizes instead of
		// the default (tight) spacing.
		"elk.spacing.nodeNode": 45,
		"elk.layered.spacing.nodeNodeBetweenLayers": 70,
		"elk.spacing.edgeNode": 25,
		"elk.layered.spacing.edgeNodeBetweenLayers": 25,
		"elk.spacing.edgeEdge": 15,

		// Straighter edges, fewer crossings, more stable ordering
		// across re-layouts (helps when you add/remove nodes live).
		"elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
		"elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",

		// Trims empty whitespace left over after layering.
		"elk.layered.compaction.postCompaction.strategy": "EDGE_LENGTH",
	},
};


function nodeSize(incoming: number): number {
	return 46 + Math.sqrt(incoming) * 12;
}

export default function GraphView(props: GraphViewProps) {
	let container!: HTMLDivElement;
	let searchInput!: HTMLInputElement;
	const confirm = useConfirm();
	const [cy, setCy] = createSignal<Core>();
	const [searchOpen, setSearchOpen] = createSignal(false);
	const [searchTerm, setSearchTerm] = createSignal("");
	const [hiddenTypes, setHiddenTypes] = createSignal<Set<string>>(new Set());
	const [filterOpen, setFilterOpen] = createSignal(false);

	function toggleType(type: string) {
		setHiddenTypes((prev) => {
			const next = new Set(prev);
			next.has(type) ? next.delete(type) : next.add(type);
			return next;
		});
	}

	function buildElements(): ElementDefinition[] {
		const hidden = hiddenTypes();
		const visibleEntities = props.entities.filter((entity) => !hidden.has(entity.type));
		const visibleIds = new Set(visibleEntities.map((entity) => entity.id));

		const visibleRelations = props.relations.filter(
			(relation) => visibleIds.has(relation.sourceId) && visibleIds.has(relation.targetId),
		);

		const incomingCounts = new Map<string, number>();
		for (const relation of [...visibleRelations].sort((a, b) => a.id.localeCompare(b.id))) {
			incomingCounts.set(relation.targetId, (incomingCounts.get(relation.targetId) ?? 0) + 1);
		}

		const nodes: ElementDefinition[] = [...visibleEntities]
			.sort((a, b) => a.id.localeCompare(b.id))
			.map((entity) => {
				const incoming = incomingCounts.get(entity.id) ?? 0;
				return {
					data: { id: entity.id, label: entity.label, type: entity.type, incoming, size: nodeSize(incoming) },
				};
			});

		const edges: ElementDefinition[] = [...visibleRelations]
			.sort((a, b) => a.id.localeCompare(b.id))
			.map((relation) => ({
				data: { id: relation.id, source: relation.sourceId, target: relation.targetId, label: relation.type },
			}));

		return [...nodes, ...edges];
	}


	function syncElements(instance: Core) {
		const elements = buildElements();

		const wantedIds = new Set(elements.map((element) => String(element.data?.id)));

		instance.elements().forEach((element) => {
			if (!wantedIds.has(element.id())) {
				element.remove();
			}
		});

		for (const element of elements) {
			const id = String(element.data?.id);
			const existing = instance.getElementById(id);

			if (existing.length > 0) {
				existing.data(element.data);
			} else {
				instance.add(element);
			}
		}
	}

	onMount(() => {
		const instance = cytoscape({
			container,
			elements: buildElements(),
			style: [
				{
					selector: "node",
					style: {
						label: "data(label)",
						"text-valign": "center",
						"text-halign": "center",

						"background-color": "#37474f",
						color: "#ffffff",
						"border-width": 2,
						"border-color": "#78909c",
						"font-size": "12px",
						"font-weight": 500,
						"text-wrap": "wrap",
						"text-max-width": "80px",

						width: "data(size)",
						height: "data(size)",
					},
				},

				...Object.keys({ ...typeColors, ...Object.fromEntries(props.entities.map((e) => [e.type, true])) }).map((type) => ({
					selector: `node[type = "${ type }"]`,
					style: {
						"background-color": `hsl(${ hueForType(type) }, 94%, 32%)`,
						"border-color": `hsl(${ hueForType(type) }, 35%, 68%)`,
					},
				})),
				{
					selector: "node:selected",
					style: {
						"background-color": "#10063f",
						"border-width": 3,
						"border-color": "#ffffff",
						color: "#ffffff",
						"overlay-color": "#26084d",
						"overlay-opacity": 0.08,
					},
				},

				{
					selector: "edge",
					style: {
						width: 2,

						"line-color": "#90a4ae",

						"target-arrow-color": "#90a4ae",
						"target-arrow-shape": "triangle",

						"curve-style": "bezier",

						label: "data(label)",

						color: "#eeeeee",
						"text-background-color": "#263238",
						"text-background-opacity": 1,
						"text-background-padding": "3px",

						"font-size": "10px",
						"font-weight": 500,
					},
				},

				{
					selector: "edge:selected",
					style: {
						width: 3,

						"line-color": "#ffffff",
						"target-arrow-color": "#ffffff",
						color: "#ffffff",
						"text-background-color": "#37474f",
					},
				},

				{
					selector: "node.link-source",
					style: {
						"border-width": 4,
						"border-color": "#ffffff",
						"overlay-color": "#ffffff",
						"overlay-opacity": 0.15,
					},
				},

				{
					selector: "node.link-target",
					style: {
						"border-width": 3,
						"border-color": "#ffffff",
					},
				},

				{
					selector: "node.search-dim",
					style: {
						opacity: 0.25,
					},
				},
				{
					selector: "node.search-match",
					style: {
						"border-width": 5,
						"border-color": "#ffffff",
						"background-color": "#10063f",
						opacity: 1,
					},
				},
			],

			layout: LAYOUT_PARAMS,
		});

		instance.on("cxttap", (event) => {
			event.originalEvent.preventDefault();

			const rect = container.getBoundingClientRect();
			const x = event.originalEvent.clientX - rect.left;
			const y = event.originalEvent.clientY - rect.top;

			if (event.target === instance) {
				setContextMenu({
					kind: "canvas",
					x,
					y,
				});

				return;
			}

			if (event.target.isNode()) {
				setContextMenu({
					kind: "node",
					x,
					y,
					nodeId: event.target.id(),
				});

				return;
			}

			if (event.target.isEdge()) {
				setContextMenu({
					kind: "edge",
					x,
					y,
					relationId: event.target.id(),
				});
			}
		});

		instance.on("tap", "edge", (event) => {
			const relationId = event.target.id();

			const relation = props.relations.find((relation) => relation.id === relationId);

			if (relation) {
				props.onSelectRelation?.(relation);
			}
		});

		instance.on("tap", "node", (event) => {
			const targetId = event.target.id();
			const sourceId = linkingFrom();

			if (sourceId) {
				if (sourceId !== targetId) {
					setLinkingFrom(undefined);

					instance.nodes().removeClass("link-target");

					instance.getElementById(sourceId).removeClass("link-source");

					const target = props.entities.find((entity) => entity.id === targetId);

					if (target) {
						props.onAddRelation?.(sourceId, targetId);
					}
				}

				return;
			}

			const entity = props.entities.find((item) => item.id === targetId);

			if (entity) {
				props.onSelectEntity?.(entity);
			}
		});

		instance.on("mouseover", "node", (event) => {
			const node = event.target;
			node.style({
				"font-size": 32,
				"font-weight": 400,
				"z-index": 9999,
			});
		});

		instance.on("mouseout", "node", (event) => {
			const node = event.target;
			node.style({
				"font-size": 12,
				"font-weight": 500,
				"z-index": 0,
			});
		});

		instance.on("mouseover", "edge", (event) => {
			const edge = event.target;
			edge.style({
				"font-size": 32,
				"font-weight": 600,
				"z-index": 999999,
			});
		});

		instance.on("mouseout", "edge", (event) => {
			const edge = event.target;
			edge.style({
				"font-size": 10,
				"font-weight": 500,
				"z-index": 0,
			});
		});

		setCy(instance);

		const handleKeyDown = (event: KeyboardEvent) => {
			if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "f") {
				event.preventDefault();
				setSearchOpen(true);
				requestAnimationFrame(() => {
					searchInput?.focus();
					searchInput?.select();
				});
				return;
			}

			if (event.key !== "Escape") return;
			setContextMenu(undefined);

			if (searchOpen()) {
				setSearchOpen(false);
				setSearchTerm("");
				return;
			}

			const sourceId = linkingFrom();
			if (sourceId) {
				setLinkingFrom(undefined);
				instance.getElementById(sourceId).removeClass("link-source");
				instance.nodes().removeClass("link-target");
			}
		};

		window.addEventListener("keydown", handleKeyDown);
		onCleanup(() => window.removeEventListener("keydown", handleKeyDown));
	});

	createEffect(() => {
		const instance = cy();
		if (!instance) return;
		syncElements(instance);
	});

	createEffect(() => {
		const instance = cy();
		const term = searchTerm().trim().toLocaleLowerCase();
		if (!instance) return;

		instance.nodes().removeClass("search-match search-dim");
		if (!term) return;

		instance.nodes().forEach((node) => {
			const label = String(node.data("label") ?? "").toLocaleLowerCase();

			if (label.includes(term)) {
				node.addClass("search-match");
			} else {
				node.addClass("search-dim");
			}
		});
	});

	onCleanup(() => cy()?.destroy());

	return (
		<div
			ref={container}
			style={{
				position: "relative",
				width: "100%",
				height: "100%",
				"min-height": "500px",
			}}
			onClick={() => setContextMenu(undefined)}
		>
			<Show when={contextMenu()}>
				{(menu) => (
					<div
						class="graph-context-menu"
						style={{
							position: "absolute",
							left: `${ menu().x }px`,
							top: `${ menu().y }px`,
							"z-index": 100000,
						}}
						onClick={(event) => event.stopPropagation()}
					>
						<menu class="active group no-wrap small-space top">
							<Switch>
								<Match when={menu().kind === "canvas"}>
									<button
										type="button"
										class="fill"
										onClick={() => {
											const item = menu();

											if (item.kind !== "canvas") {
												return;
											}

											props.onAddEntity?.({
												x: item.x,
												y: item.y,
											});

											setContextMenu(undefined);
										}}
									>
										Add node
									</button>
								</Match>

								<Match when={menu().kind === "node"}>
									<button
										type="button"
										class="fill"
										onClick={() => {
											const item = menu();

											if (item.kind !== "node") {
												return;
											}

											const entity = props.entities.find((entity) => entity.id === item.nodeId);

											if (entity) {
												props.onEditEntity?.(entity);
											}

											setContextMenu(undefined);
										}}
									>
										Edit node
									</button>

									<button
										type="button"
										class="fill"
										onClick={() => {
											const item = menu();
											if (item.kind !== "node") {
												return;
											}
											setLinkingFrom(item.nodeId);
											const instance = cy()!;

											instance.getElementById(item.nodeId).addClass("link-source");
											instance
												.nodes()
												.not(`#${ CSS.escape(item.nodeId) }`)
												.addClass("link-target");

											setContextMenu(undefined);
										}}
									>
										Add relation →
									</button>

									<button
										type="button"
										class="error-container on-error"
										onClick={() => {
											const item = menu();

											if (item.kind !== "node") {
												return;
											}

											const entity = props.entities.find((entity) => entity.id === item.nodeId);

											if (entity) {
												props.onDeleteEntity?.(entity);
											}

											setContextMenu(undefined);
										}}
									>
										Delete node
									</button>
								</Match>

								<Match when={menu().kind === "edge"}>
									<button
										type="button"
										onClick={() => {
											const item = menu();

											if (item.kind !== "edge") {
												return;
											}

											const relation = props.relations.find((relation) => relation.id === item.relationId);

											if (relation) {
												props.onEditRelation?.(relation);
											}

											setContextMenu(undefined);
										}}
									>
										Edit relation
									</button>

									<button
										type="button"
										class="danger"
										onClick={async () => {
											const item = menu();
											if (item.kind !== "edge") {
												return;
											}
											const ok = await confirm(`Delete this relation?`);
											if (!ok) return;

											const relation = props.relations.find((relation) => relation.id === item.relationId);

											if (relation) {
												props.onDeleteRelation?.(relation);
											}

											setContextMenu(undefined);
										}}
									>
										Delete relation
									</button>
								</Match>
							</Switch>
						</menu>
					</div>
				)}
			</Show>

			<Show when={searchOpen()}>
				<div
					class="graph-search"
					onClick={(event) => event.stopPropagation()}
					style={{
						position: "absolute",
						top: "1em",
						left: "50%",
						transform: "translateX(-50%)",
						"z-index": 100000,
					}}
				>
					<nav class="no-space">
						<div class="max field border left-round">
							<input
								ref={searchInput}
								type="text"
								placeholder="Find a node…"
								value={searchTerm()}
								onInput={(event) => setSearchTerm(event.currentTarget.value)}
								style={{ width: "20em" }}
							/>
						</div>
						{/* <Show when={searchTerm().trim()}>
								<span>{cy()?.nodes(".search-match").length ?? 0}</span>
							</Show> */}
						<button
							type="button"
							class="large right-round"
							onClick={() => {
								setSearchOpen(false);
								setSearchTerm("");
							}}
							title="Close search"
						>
							<i>close</i>
						</button>
					</nav>
				</div>
			</Show>

			<Show when={linkingFrom()}>
				<div
					class="graph-linking-indicator fill padding round"
					style={{
						position: "absolute",
						bottom: "2em",
						left: "50%",
						transform: "translateX(-50%)",
						"z-index": 100000,
					}}
				>
					Click a node to create the relation. Press Escape to cancel.
				</div>
			</Show>

			<button
				type="button"
				onClick={() => setFilterOpen((v) => !v)}
				class="circle tertiary"
				style="position: absolute; bottom: 1rem; right: 1rem"
				title="Filter node types"
			>
				<i>filter_alt</i>
			</button>

			<Show when={true || filterOpen()}>
				<div
					class="graph-filter-panel surface padding round"
					onClick={(event) => event.stopPropagation()}
					style={{
						position: "absolute",
						right: "1rem",
						bottom: "4rem",
						"z-index": 100000,
						"min-width": "13rem",
					}}
				>
					<For each={[...new Set(props.entities.map((e) => e.type))].sort()}>
						{(type) => {
							const count = () => props.entities.filter((e) => e.type === type).length;
							return (
								<label style={{ display: "flex", "flex-direction": "row-reverse", "align-items": "center", gap: "0.5em", padding: "0.25em 0" }}>
									<input
										type="checkbox"
										checked={!hiddenTypes().has(type)}
										onChange={() => toggleType(type)}
									/>
									<span
										style={{
											display: "inline-block",
											width: "10pt",
											height: "10pt",
											"border-radius": "50%",
											"background-color": `hsl(${ hueForType(type) }, 94%, 32%)`,
										}}
									/>
									<span>{type} ({count()})</span>
								</label>
							);
						}}
					</For>
				</div>
			</Show>
		</div>
	);
}
