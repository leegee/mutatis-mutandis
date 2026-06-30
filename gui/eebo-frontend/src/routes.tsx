import { lazy } from "solid-js";

import { JobsApiComponent } from "./components/Jobs";
const Graph2 = lazy(() => import("./components/Graph2/Graph2"));
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

export const routes = [
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
    path: "/geo",
    icon: "map",
    label: "Geography",
    component: Map,
    help: () => { },
  },
  {
    path: "/graph2/:token_idx?",
    icon: "orbit",
    label: "FDG",
    component: Graph2,
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
    path: "/Jobs",
    icon: "api",
    label: "Job Admin",
    component: JobsApiComponent,
    help: () => <div>Temporal evolution of concepts.</div>
  }

] as const;
