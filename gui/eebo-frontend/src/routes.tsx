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
const ConceptClusters = lazy(() => import("./components/ConceptClusters"));

export const routes = [
  {
    path: "/scatter",
    icon: "scatter_plot",
    label: "Scatter Plot",
    component: ScatterPlot,
    help: () => (
      <section>
        <p>
          The scatter plot shows embedding relationships between concepts
          through UMP and PACMAP collapsisings of 768 dimensions of real numbers
          to integers in a two-dimensional space.
        </p>
        <p>
          Select a point by clicking, and also both <kbd>SHIFT</kbd>-clicking and clicking then dragging whilst holding <kbd>SHIFT</kbd> key.
          Once you have thus raised summary reports of the selected mini-corpus, you may wish to down your selection,
          or copy it to your clipboard. Several styles of export are available.
        </p>
        <p>
          Other options are exploratory prototypes - send parts of selections to <code>llama-3.3-70b-versatile</code>
          to see its attempt at classification which really needs luca to unlock larger payloads that can carry multiople clusters,
          as a differential task might yield better results than one where there is no supplied refereant but where the frame of
          reference is solely in the prompts.
        </p>
        <p>
          <em>TODO</em> &mdash; let's add a new layer with opacity+visibility controls, to hold results GPT definitions of clusters - anchor at centroid of same.
        </p>
      </section>
    )
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
