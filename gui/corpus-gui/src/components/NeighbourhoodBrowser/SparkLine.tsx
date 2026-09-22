interface Props {
  color?: string;
  data: {
    year: number;
    value: number
  }[]
}

export default function Sparkline(props: Props) {
  const width = 60;
  const height = 14;

  const max = Math.max(...props.data.map(d => d.value), 1);
  const minYear = props.data[0]?.year ?? 0;
  const maxYear = props.data[props.data.length - 1]?.year ?? 1;

  const points = props.data.map(d => {
    const x = ((d.year - minYear) / (maxYear - minYear || 1)) * width;
    const y = height - (d.value / max) * height;
    return `${ x },${ y }`;
  }).join(" ");

  return (
    <svg width={width} height={height} style={{ opacity: 0.7 }}>
      <polyline
        points={points}
        fill="none"
        stroke={props.color || "currentColor"}
        stroke-width="2"
      />
    </svg>
  );
}
