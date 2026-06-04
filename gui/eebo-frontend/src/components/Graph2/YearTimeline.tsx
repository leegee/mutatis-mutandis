import { type Component, For } from "solid-js";

export interface YearBucket {
  year: number;
  count: number;
}

interface YearTimelineProps {
  years: YearBucket[];

  yearMode: "single" | "range";

  fromYear: number;
  toYear: number;

  onSelect?: (year: number) => void;
}

export const YearTimeline: Component<YearTimelineProps> = (props) => {
  const maxCount = () =>
    Math.max(...props.years.map(y => y.count), 1);

  const isSelected = (year: number) => {
    if (props.yearMode === "single") {
      return year === props.fromYear;
    }

    return (
      year >= props.fromYear &&
      year <= props.toYear
    );
  };

  return (
    <div
      class="row center-align small-padding"
      style={{
        gap: "2px",
        height: "44px",
        "align-items": "flex-end",
      }}
    >
      <For each={props.years}>
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
            <button
              class={`no-border transparent ${ selected() ? "primary-container" : ""
                }`}
              title={`${ bucket.year } (${ bucket.count } events)`}
              onClick={() => props.onSelect?.(bucket.year)}
              style={{
                padding: "0",
                width: "12px",
                height: "40px",
                display: "flex",
                "align-items": "flex-end",
                "justify-content": "center",
                cursor: "pointer",
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
            </button>
          );
        }}
      </For>
    </div>
  );
};
