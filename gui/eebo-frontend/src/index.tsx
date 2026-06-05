import { lazy } from "solid-js";
import { render } from "solid-js/web";
import { Router, Route } from "@solidjs/router";
import "beercss";

import "./index.css";

import App from "./App";

const Graph2 = lazy(() => import("./components/Graph2/Graph2"));
const NeighbourhoodBrowser = lazy(() => import("./components/NeighbourhoodBrowser"));
const DiachronicChart = lazy(() => import("./components/DiachronicChart"));
const Umap = lazy(() => import("./components/Umap"));

render(
    () => (
        <Router root={App}>
            <Route path="/" component={Umap} />
            <Route path="/umap" component={Umap} />
            <Route path="/graph2/:token_idx?" component={Graph2} />
            <Route path="/table" component={NeighbourhoodBrowser} />
            <Route path="/diachronic" component={DiachronicChart} />
        </Router>
    ),
    document.getElementById("root")!
);