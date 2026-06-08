import { lazy } from "solid-js";
import { render } from "solid-js/web";
import { Router, Route } from "@solidjs/router";
import "beercss";

import "./index.css";

import App from "./App";

const Graph2 = lazy(() => import("./components/Graph2/Graph2"));
const NeighbourhoodBrowser = lazy(() => import("./components/NeighbourhoodBrowser"));
const DiachronicChart = lazy(() => import("./components/DiachronicChart"));
const ScatterPlot = lazy(() => import("./components/ScatterPlot"));
const ConceptAggregates = lazy(() => import("./components/ConceptAggregates"));

render(
    () => (
        <Router root={App}>
            <Route path="/" component={ScatterPlot} />
            <Route path="/umap" component={ScatterPlot} />
            <Route path="/graph2/:token_idx?" component={Graph2} />
            <Route path="/table" component={NeighbourhoodBrowser} />
            <Route path="/diachronic" component={DiachronicChart} />
            <Route path="/aggregates" component={ConceptAggregates} />
        </Router>
    ),
    document.getElementById("root")!
);