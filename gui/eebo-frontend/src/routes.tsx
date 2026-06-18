import { lazy } from "solid-js";

const Graph2 = lazy(() => import("./components/Graph2/Graph2"));
const NeighbourhoodBrowser = lazy(() => import("./components/NeighbourhoodBrowser/NeighbourhoodBrowser"));
const NeighbourhoodBrowserGuide = lazy(() => import("./components/NeighbourhoodBrowser/Guide"));
const DiachronicChart = lazy(() => import("./components/DiachronicChart/DiachronicChart"));
const ScatterPlot = lazy(() => import("./components/ScatterPlot"));
const ConceptAggregates = lazy(() => import("./components/ConceptAggregates"));
const ConceptClusters = lazy(() => import("./components/ConceptClusters"));

export const routes = [
  {
    path: "/scatter",
    icon: "scatter_plot",
    label: "Scatter Plot",
    component: ScatterPlot,
    help: () => (
      <div>
        Scatter plot shows embedding relationships between concepts.
      </div>
    )
  },
  {
    path: "/aggregates",
    icon: "crowdsource",
    label: "Aggregates",
    component: ConceptAggregates,
    help: () => (
      <div>
        Aggregates show grouped concept statistics.
      </div>
    )
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
    help: () => <div>Temporal evolution of concepts.</div>
  }
] as const;
