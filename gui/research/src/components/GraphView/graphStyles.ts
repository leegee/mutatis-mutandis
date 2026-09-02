import type { StylesheetCSS } from "cytoscape";

import type { Entity } from "~/domain/entity";
import { hueForType, typeColors } from "./clrs";

export const graphStyles = (entities: Entity[]): StylesheetCSS[] => {
	return [
		{
			selector: "node",
			css: {
				label: "data(label)",
				"text-valign": "center",
				"text-halign": "center",
				"background-color": "#37474f",
				color: "#ffffff",
				"border-width": 2,
				"border-color": "#78909c",
				"font-size": "14px",
				"font-weight": 500,
				"text-wrap": "wrap",
				"text-max-width": "80px",
				width: "data(size)",
				height: "data(size)",
			},
		},

		...Object.keys({ ...typeColors, ...Object.fromEntries(entities.map((e) => [e.type, true])) }).map((type) => ({
			selector: `node[type = "${type}"]`,
			css: {
				"background-color": `hsl(${hueForType(type)}, 94%, 32%)`,
				"border-color": `hsl(${hueForType(type)}, 35%, 68%)`,
			},
		})),
		{
			selector: "node:selected",
			css: {
				"font-size": "48px",
				"background-color": "#10063f",
				"text-background-color": "#37474f",
				"border-width": 3,
				"border-color": "#ffffff",
				color: "#ffffff",
				"text-max-width": "30em",
				"overlay-color": "#26084d",
				"overlay-opacity": 0.08,
			},
		},
		{
			selector: "node.hovered",
			css: {
				"font-size": 32,
				"font-weight": 600,
				"z-index": 9999,
			},
		},

		{
			selector: "edge",
			css: {
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

				"font-size": "15px",
				"font-weight": 500,
			},
		},

		{
			selector: "edge.selected-connected",
			css: {
				width: 4,
				"line-color": "#ffffff",
				"target-arrow-color": "#ffffff",
				color: "#ffffff",
				"text-background-color": "#37474f",
			},
		},

		{
			selector: "edge:selected",
			css: {
				width: 3,

				"line-color": "#ffffff",
				"target-arrow-color": "#ffffff",
				color: "#ffffff",
				"text-background-color": "#37474f",
			},
		},

		{
			selector: "node.link-source",
			css: {
				"border-width": 4,
				"border-color": "#ffffff",
				"overlay-color": "#ffffff",
				"overlay-opacity": 0.15,
			},
		},

		{
			selector: "edge.hover-connected",
			css: {
				width: 4,
				"line-color": "#ffffff",
				"target-arrow-color": "#ffffff",
				color: "#ffffff",
				"text-background-color": "#37474f",
			},
		},
		{
			selector: "edge.hover-unconnected",
			css: {
				opacity: 0.8,
			},
		},

		{
			selector: "node.link-target",
			css: {
				"border-width": 3,
				"border-color": "#ffffff",
			},
		},

		{
			selector: "node.filtered-out, edge.filtered-out",
			css: {
				display: "none",
			},
		},

		{
			selector: "node.search-dim",
			css: {
				opacity: 0.25,
			},
		},
		{
			selector: "edge.search-dim",
			css: {
				opacity: 0.25,
			},
		},

		{
			selector: "node.search-match",
			css: {
				"border-width": 5,
				"border-color": "#ffffff",
				"background-color": "#10063f",
				opacity: 1,
				"font-size": "2em",
				"text-max-width": "30em",
			},
		},
	] as StylesheetCSS[];
};
