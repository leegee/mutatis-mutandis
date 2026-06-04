import { type Component, createResource, For } from "solid-js";
import { getYearBuckets } from "../state/selectors";
import { controls } from "../state/controls.store";
import { controlsActions as A } from "../state/controls.actions";

interface YearTimelineProps {
  tooltipPosition?: 'top' | 'bottom' | null;
  onSelect?: (year: number) => void;
}

export const YearTimeline: Component<YearTimelineProps> = (props) => {
  const [yearBucketsResource] = createResource(
    () => controls.concept,
    (concept) => getYearBuckets(concept),
  );

  const maxCount = () => Math.max(...(yearBucketsResource() ?? []).map(y => y.count), 1);

  const isSelected = (year: number) => {
    if (controls.yearMode === "single") {
      return year === controls.fromYear;
    }

    return (
      year >= controls.fromYear &&
      year <= controls.toYear
    );
  };

  return (
    <aside class="surface-container row center-align small-padding" style={{ gap: "0.5pt", }} >

      <button class="circle chip tiny no-border" onClick={() => A.stepYear(-1)} >
        <i>chevron_left</i>
        <span class="tooltip bottom">Retreat by one year</span>
      </button>

      <For each={yearBucketsResource()}>
        {(bucket) => {
          const selected = () => isSelected(bucket.year);

          const height = () =>
            Math.max(
              4,
              Math.round(
                (bucket.count / maxCount()) * 28,
              ),
            );

          return (
            <button class={`no-border no-padding transparent ${ selected() ? "tertiary-container" : ""
              }`}
              onClick={() => {
                A.setSingleYear(bucket.year);
              }}
              style={{
                width: "12px",
                height: "40px",
                display: "flex",
                "align-items": "flex-end",
                "justify-content": "center",
                cursor: "crosshair",
              }}
            >
              <div
                style={{
                  width: "8px",
                  height: `${ height() }px`,
                  "border-radius": "2px",
                  opacity: bucket.count === 0 ? 0.15 : 1,
                  background:
                    bucket.count === 0
                      ? "var(--outline)"
                      : selected()
                        ? "var(--primary)"
                        : "var(--secondary)",
                }}
              />
              <div class={`tooltip ${ props.tooltipPosition || 'top' }`}>
                <span class="bold">{bucket.year} </span>
                &mdash;
                {bucket.count} events
              </div>
            </button>
          );
        }}
      </For>

      <button class="circle chip tiny no-border" onClick={() => A.stepYear(1)} >
        <i>chevron_right</i>
        <span class="tooltip bottom">Advance by one year</span>
      </button>

      <button class="circle chip tiny no-border small-text no-line" style="font-size:0.5rem"
        onClick={A.setAllYears}
      >
        ALL YEARS
        <span class="tooltip bottom">Show all years</span>
      </button>


    </aside >
  );
};
