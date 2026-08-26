/** biome-ignore-all lint/a11y/useKeyWithClickEvents: Time is limited now */
/** biome-ignore-all lint/a11y/noStaticElementInteractions: Time is limited now */

import cytoscape, { type Core, type ElementDefinition } from "cytoscape";
import cytoscapeElk from "cytoscape-elk";
import { createEffect, createSignal, For, Match, onCleanup, onMount, Show, Switch } from "solid-js";

import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

import { hueForType } from "./GraphView/clrs";
import { graphStyles } from "./GraphView/graphStyles";
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

const NODE_SCALE_FACTOR = () => 50;

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
	return 46 + Math.sqrt(incoming) * NODE_SCALE_FACTOR();
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
		const incomingCounts = new Map<string, number>();

		for (const relation of [...props.relations].sort((a, b) => a.id.localeCompare(b.id))) {
			incomingCounts.set(relation.targetId, (incomingCounts.get(relation.targetId) ?? 0) + 1);
		}

		const nodes: ElementDefinition[] = [...props.entities]
			.sort((a, b) => a.id.localeCompare(b.id))
			.map((entity) => {
				const incoming = incomingCounts.get(entity.id) ?? 0;

				return {
					data: {
						id: entity.id,
						label: entity.label,
						type: entity.type,
						incoming,
						size: nodeSize(incoming),
					},
				};
			});

		const edges: ElementDefinition[] = [...props.relations]
			.sort((a, b) => a.id.localeCompare(b.id))
			.filter(
				(relation) =>
					props.entities.some((entity) => entity.id === relation.sourceId) &&
					props.entities.some((entity) => entity.id === relation.targetId),
			)
			.map((relation) => ({
				data: {
					id: relation.id,
					source: relation.sourceId,
					target: relation.targetId,
					label: relation.type,
				},
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
			style: graphStyles(props.entities),
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

		instance.on("select", "node", (event) => {
			const node = event.target;
			instance.edges().removeClass("selected-connected");
			node.connectedEdges().addClass("selected-connected");
		});

		instance.on("unselect", "node", () => {
			instance.edges().removeClass("selected-connected");
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
			node.addClass("hovered");

			instance.edges().removeClass("hover-connected hover-unconnected");

			instance.edges().forEach((edge) => {
				if (edge.source().id() === node.id() || edge.target().id() === node.id()) {
					edge.addClass("hover-connected");
				} else {
					edge.addClass("hover-unconnected");
				}
			});
		});

		instance.on("mouseout", "node", (event) => {
			event.target.removeClass("hovered");
			instance.edges().removeClass("hover-connected hover-unconnected");
		});

		instance.on("mouseover", "edge", (event) => {
			const edge = event.target;
			edge.addClass("hovered");
			instance.edges().not(edge).removeClass("hovered");
		});

		instance.on("mouseout", "edge", (event) => {
			event.target.removeClass("hovered");
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

	// User filters hide graph elements
	createEffect(() => {
		const hidden = hiddenTypes();
		const instance = cy();
		if (!instance) return;

		instance.nodes().forEach((node) => {
			node.toggleClass("filtered-out", hidden.has(node.data("type")));
		});

		instance.edges().forEach((edge) => {
			const sourceHidden = edge.source().hasClass("filtered-out");
			const targetHidden = edge.target().hasClass("filtered-out");

			edge.toggleClass("filtered-out", sourceHidden || targetHidden);
		});
	});

	createEffect(() => {
		const instance = cy();
		const term = searchTerm().trim().toLocaleLowerCase();
		if (!instance) return;

		instance.nodes().removeClass("search-match search-dim");
		instance.edges().removeClass("search-dim");

		if (!term) return;

		const matchingNodes = instance.nodes().filter((node) => {
			const label = String(node.data("label") ?? "").toLocaleLowerCase();
			return label.includes(term);
		});

		matchingNodes.addClass("search-match");
		instance.nodes().not(matchingNodes).addClass("search-dim");

		instance.edges().forEach((edge) => {
			const sourceMatches = matchingNodes.contains(edge.source());
			const targetMatches = matchingNodes.contains(edge.target());

			if (!sourceMatches && !targetMatches) {
				edge.addClass("search-dim");
			}
		});
	});

	onCleanup(() => cy()?.destroy());

	return (
		<>
			<div
				ref={container}
				style={{
					position: "relative",
					width: "100%",
					height: "100%",
					"min-height": "500px",
				}}
				onClick={() => setContextMenu(undefined)}
			></div>

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
											if (item.kind !== "canvas") return;

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

											if (item.kind !== "node") return;

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
											if (item.kind !== "node") return;
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
											if (item.kind !== "node") return;

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
											if (item.kind !== "edge") return;

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
											if (item.kind !== "edge") return;

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
					<nav class="no-space top-margin">
						<div class="max field border round fill">
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
							class="left-margin right-margin circle transparent border small"
							onClick={() => {
								setSearchOpen(false);
								setSearchTerm("");
							}}
							title="Close search"
						>
							<i>close</i>
							<span class="tooltip right">Close the search</span>
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
				<Show when={filterOpen()}>
					<i> arrow_drop_down</i>
					<span class="tooltip left">Close visible node filter</span>
				</Show>
				<Show when={!filterOpen()}>
					<i> filter_alt</i>
					<span class="tooltip left">Open visible node filter</span>
				</Show>
			</button >

			<Show when={filterOpen()}>
				<div
					class="graph-filter-panel surface padding round elevate border small-round right-align"
					onClick={(event) => event.stopPropagation()}
					style={{
						position: "absolute",
						right: "0rem",
						bottom: "4rem",
						"z-index": 10,
						"min-width": "10em",
					}}
				>
					<For each={[...new Set(props.entities.map((e) => e.type))].sort()}>
						{(type) => {
							const count = () => props.entities.filter((e) => e.type === type).length;
							return (
								<label style={{
									display: "flex",
									"flex-direction": "row-reverse",
									"align-items": "center",
									gap: "0.5em",
									padding: "0.25em 0",
								}}
								>
									<input type="checkbox" checked={!hiddenTypes().has(type)} onChange={() => toggleType(type)} />
									<span style={{
										display: "inline-block",
										width: "1em",
										height: "1em",
										"border-radius": "50%",
										"background-color": `hsl(${ hueForType(type) }, 94%, 52%)`,
									}}
									/>
									<span>
										{type} ({count()})
									</span>
								</label>
							);
						}}
					</For>
				</div>
			</Show>
		</>
	);
}
