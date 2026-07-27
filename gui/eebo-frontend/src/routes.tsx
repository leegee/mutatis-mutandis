import { lazy, type Component } from "solid-js";

import { JobsApiComponent } from "./components/Jobs";
import Home from "./components/Home";
const FDG = lazy(() => import("./components/FDG"));
const NeighbourhoodBrowser = lazy(() => import("./components/NeighbourhoodBrowser/NeighbourhoodBrowser"));
const NeighbourhoodBrowserGuide = lazy(() => import("./components/NeighbourhoodBrowser/Guide"));
const DiachronicChart = lazy(() => import("./components/DiachronicChart/DiachronicChart"));
const DiachronicChartGuide = lazy(() => import("./components/DiachronicChart/Guide"));
const ScatterPlot = lazy(() => import("./components/ScatterPlot"));
const ConceptAggregates = lazy(() => import("./components/ConceptAggregates"));
import ConceptAggregatesGuide from "./components/ConceptAggregatesGuide";
import Map from "./components/GeoMap";
import ScatterPlotGuide from "./components/ScatterPlot/ScatterPlotGuide";
const ConceptClusters = lazy(() => import("./components/ConceptClusterReport/ConceptClusters"));
const LineageGraph = lazy(() => import("./components/LineageGraph"));

export interface RouteType {
  path: string;
  icon: string;
  label: string;
  component: Component;
  help?: Component | undefined;
}

export const routes = [
  {
    path: "/",
    icon: "home",
    label: "Home",
    component: Home,
  },

  {
    path: "/lineage",
    icon: "family_history",
    label: "Lineage",
    component: LineageGraph,
  },

  {
    path: "/scatter",
    icon: "scatter_plot",
    label: "Scatter Plot",
    component: ScatterPlot,
    help: () => <ScatterPlotGuide />
  },
  {
    path: "/aggregates",
    icon: "crowdsource",
    label: "Aggregates",
    component: ConceptAggregates,
    help: () => <ConceptAggregatesGuide />
  },
  {
    path: "/clusters",
    icon: "action_key",
    label: "Cluster Report",
    component: ConceptClusters,
    help: () => (
      <div>
        Cluster report summarizes grouped structure.
      </div>
    )
  },
  {
    path: "/graph2/:token_idx?",
    icon: "orbit",
    label: "FDG",
    component: FDG,
    help: () => <div>Force-directed graph exploration view.</div>
  },
  {
    path: "/table",
    icon: "tenancy",
    label: "Neighbourhood Browser",
    component: NeighbourhoodBrowser,
    help: () => <NeighbourhoodBrowserGuide />
  },
  {
    path: "/diachronic",
    icon: "chronic",
    label: "Diachronic Chart",
    component: DiachronicChart,
    help: () => <DiachronicChartGuide />
  },
  {
    path: "/geo",
    icon: "location_on",
    label: "Geography",
    component: Map,
    help: () => { },
  },
  {
    path: "/Jobs",
    icon: "api",
    label: "Job Admin",
    component: JobsApiComponent,
    help: () => <div>Temporal evolution of concepts.</div>
  }

] as RouteType[];
