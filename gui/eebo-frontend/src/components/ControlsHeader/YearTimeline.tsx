import { type Component, createResource, For, onCleanup, onMount } from "solid-js";
import { controlsActions as A } from "../../state/controls.actions";
import { getYearBuckets } from "../../state/controls.selectors";
import { controls } from "../../state/controls.store";

import "./YearTimeline.css";
import { Portal } from "solid-js/web";

export const YearTimeline: Component = () => {
  const [yearBucketsResource] = createResource(
    () => controls.conceptSelection[0],
    (concept) => getYearBuckets(concept),
  );

  const maxCount = () => Math.max(...(yearBucketsResource() ?? []).map((y) => y.count), 1);

  const isSelected = (year: number) => {
    if (controls.yearMode === "single") {
      return year === controls.fromYear;
    }

    return year >= controls.fromYear && year <= controls.toYear;
  };

  function handleKeyDown(e: KeyboardEvent) {
    // Ignore if typing into a control
    const target = e.target as HTMLElement | null;
    if (
      target &&
      (target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.tagName === "SELECT" ||
        target.isContentEditable)
    ) {
      return;
    }

    if (!e.ctrlKey) return;

    switch (e.key) {
      case "ArrowLeft":
        if (controls.fromYear === controls.toYear) {
          e.preventDefault();
          A.stepYear(-1);
        }
        break;

      case "ArrowRight":
        if (controls.fromYear === controls.toYear) {
          e.preventDefault();
          A.stepYear(1);
        }
        break;

      case "Enter":
        e.preventDefault();
        A.setAllYears();
        break;
    }
  }

  onMount(() => {
    window.addEventListener("keydown", handleKeyDown);

    if (window) onCleanup(() => window.removeEventListener("keydown", handleKeyDown));
  });

  return (
    <>
      <aside class="year-timeilne surface-container row center-align small-padding">
        <button type="button"
          class="circle chip tiny no-border"
          onClick={() => A.stepYear(-1)}
          disabled={controls.fromYear !== controls.toYear}
        >
          <i>chevron_left</i>
          <span class="tooltip bottom">
            Retreat by one year
            <kbd class="medium-opacity">CTRL ←</kbd>
          </span>
        </button>

        <span class="year-buttons">
          <For each={yearBucketsResource()}>
            {(bucket) => {
              const selected = () => isSelected(bucket.year);

              const height = () => Math.max(4, Math.round((bucket.count / maxCount()) * 100));

              return (
                <button type="button"
                  class={`year no-border no-padding transparent ${ selected() ? "tertiary-container" : "" }`}
                  title={`${ bucket.year } - ${ bucket.count }`}
                  onClick={(e) => {
                    if (e.shiftKey) {
                      if (selected()) {
                        if (controls.yearMode === "single") {
                          // no-op — can't shrink a single year
                        } else if (bucket.year === controls.fromYear) {
                          A.setSingleYear(controls.toYear);
                        } else if (bucket.year === controls.toYear) {
                          A.setSingleYear(controls.fromYear);
                        } else {
                          // Interior year: trim toward the nearer end
                          const distFrom = bucket.year - controls.fromYear;
                          const distTo = controls.toYear - bucket.year;
                          if (distFrom <= distTo) {
                            A.setRange(bucket.year + 1, controls.toYear);
                          } else {
                            A.setRange(controls.fromYear, bucket.year - 1);
                          }
                        }
                      } else {
                        // Extend to include this year
                        A.setRange(Math.min(controls.fromYear, bucket.year), Math.max(controls.toYear, bucket.year));
                      }
                    } else {
                      A.setSingleYear(bucket.year);
                    }
                  }}
                >
                  <div
                    class="year-data"
                    style={{
                      height: `${ height() }%`,
                      opacity: bucket.count === 0 ? 0.15 : 1,
                      background:
                        bucket.count === 0 ? "var(--outline)" : selected() ? "var(--primary)" : "var(--secondary)",
                    }}
                  />
                </button>
              );
            }}
          </For>
        </span>


        <button type="button"
          class="circle chip tiny no-border"
          onClick={() => A.stepYear(1)}
          disabled={controls.fromYear !== controls.toYear}
        >
          <i>chevron_right</i>
          <span class="tooltip bottom">
            Advance by one year
            <kbd class="medium-opacity">CTRL →</kbd>
          </span>
        </button>

        <button type="button"
          class="circle chip tiny no-border small-text no-line" style="font-size:0.5rem" onClick={A.setAllYears}>
          <i>all_inclusive</i>
          <span class="tooltip bottom">
            Show all years
            <kbd class="medium-opacity">CTRL ENTER</kbd>
          </span>
        </button>
      </aside>

      <Portal>
        <div class="fill round" id="giant-year-range">
          {controls.fromYear}
          {controls.toYear !== controls.fromYear ? "-" + controls.toYear : ""}
        </div>
      </Portal>
    </>
  );
};
