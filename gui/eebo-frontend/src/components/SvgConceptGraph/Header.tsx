import { For, Show, type Accessor } from "solid-js";
import { controls, setControls } from "../../state/controls";
import type { ConceptEvent, ViewMode } from "./types";
import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "../../corpus_config";

interface Props {
  includeHubSpread: boolean;
  concepts: any;
  MAX_TOP_N: number;
  yearFiltered: Accessor<ConceptEvent[]>;
  yearBounds: any;
}

export default function Header(props: Props) {
  return (

    <header class="center-align fill max surface-container-low small-padding top-padding">
      <nav>
        <div class="field suffix border middle-align">
          <select value={controls.concept}
            onChange={(e) => { setControls('concept', e.currentTarget.value); setControls('selectedNode', null); }}>
            <For each={props.concepts}>{(c) => <option value={c}>{c}</option>}</For>
          </select>
          <output>Concept</output>
        </div>

        <div class="field suffix border middle-align">
          <select value={controls.viewMode}
            onChange={(e) => { setControls('viewMode', e.currentTarget.value as ViewMode); setControls('selectedNode', null); }}>
            <option value="aggregated">Aggregated</option>
            <option value="events">Events</option>
          </select>
          <output>View</output>
        </div>

        <Show when={controls.viewMode === "aggregated"}>
          <div class="field suffix border middle-align">
            <select value={controls.maxHubs}
              onChange={(e) => setControls('maxHubs', Number(e.currentTarget.value))}>
              <For each={[10, 20, 50, 100]}>{(n) => <option value={n}>{n}</option>}</For>
            </select>
            <output>Max hubs</output>
          </div>
        </Show>

        <div class="field middle-align">
          <div class="slider tiny">
            <input type="range" min={1} max={props.MAX_TOP_N} step={1} value={controls.topN}
              onInput={(e) => setControls('topN', Number(e.currentTarget.value))} />
            <span /><span class="tooltip bottom" />
          </div>
          <output class="small-padding top-padding">Top N {controls.topN}</output>
        </div>

        <Show when={props.includeHubSpread}>
          <div class="field middle-align">
            <div class="slider tiny">
              <input type="range" min={0.2} max={2.0} step={0.05} value={controls.hubSpread}
                onInput={(e) => setControls('hubSpread', Number(e.currentTarget.value))} />
              <span /><span class="tooltip bottom" />
            </div>
            <output class="small-padding top-padding">Hub spread {controls.hubSpread.toFixed(2)}</output>
          </div>
        </Show>

        <Show when={controls.viewMode === "aggregated"}>
          <div class="field middle-align">
            <div class="slider tiny">
              <input type="range" min={0.01} max={0.95} step={0.05} value={controls.minSimilarity}
                onInput={(e) => setControls('minSimilarity', Number(e.currentTarget.value))} />
              <span /><span class="tooltip bottom" />
            </div>
            <output class="small-padding top-padding">Min sim {controls.minSimilarity.toFixed(2)}</output>
          </div>
        </Show>

        <div class="field suffix border middle-align">
          <select value={controls.yearMode}
            onChange={(e) => setControls('yearMode', e.currentTarget.value as "single" | "range")}>
            <option value="single">Single year</option>
            <option value="range">Year range</option>
          </select>
          <output>Year mode</output>
        </div>

        <Show when={controls.yearMode === "single"}>
          <nav class="no-space">
            <button class="circle chip secondary no-space large-margin bottom-margin"
              onClick={() => { const v = Math.max(CORPUS_START_YEAR, controls.fromYear - 1); setControls('fromYear', v); setControls('toYear', v); }}>
              <i>remove</i>
            </button>
            <div class="field middle-align">
              <div class="slider tiny">
                <input type="range" min={CORPUS_START_YEAR} max={CORPUS_END_YEAR} step={1}
                  value={controls.fromYear}
                  onInput={(e) => { const v = Number(e.currentTarget.value); setControls('fromYear', v); setControls('toYear', v); }} />
                <span class="tooltip bottom" />
              </div>
              <output class="small-padding top-padding">
                {controls.fromYear} ({props.yearFiltered().length} events)
              </output>
            </div>
            <button class="circle chip secondary no-space large-margin bottom-margin"
              onClick={() => { const v = Math.min(CORPUS_END_YEAR, controls.toYear + 1); setControls('toYear', v); setControls('fromYear', v); }}>
              <i>add</i>
            </button>
          </nav>
        </Show>

        <Show when={controls.yearMode === "range"}>
          <div class="field middle-align">
            <div class="slider tiny">
              <input type="range" min={props.yearBounds()[0]} max={props.yearBounds()[1]} step={1}
                value={controls.fromYear}
                onInput={(e) => setControls('fromYear', Math.min(Number(e.currentTarget.value), controls.toYear))} />
              <input type="range" min={props.yearBounds()[0]} max={props.yearBounds()[1]} step={1}
                value={controls.toYear}
                onInput={(e) => setControls('toYear', Math.max(Number(e.currentTarget.value), controls.fromYear))} />
              <span /><span class="tooltip bottom" /><span class="tooltip bottom" />
            </div>
            <output class="small-padding top-padding">
              <span>{controls.fromYear}–{controls.toYear}</span>
              {/* <span class="left-padding">{props.yearFiltered().length}
                /
                {props.data[controls.concept]?.n_events ?? 0} events
              </span> */}
            </output>
          </div>
        </Show>
      </nav>
    </header>


  )
}