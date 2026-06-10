interface Props {
  class?: string;
}

export function Icon(props: Props) {
  return (
    <svg
      width="512"
      height="512"
      viewBox="0 0 512 512"
      xmlns="http://www.w3.org/2000/svg"
      class={props.class}
    >
      <rect width="512" height="512" rx="96" fill="#0F172A" />

      <polygon
        points="256,96 384,176 384,336 256,416 128,336 128,176"
        fill="none"
        stroke="#38BDF8"
        stroke-width="18"
        stroke-linejoin="round"
      />

      <g stroke="#E2E8F0" stroke-width="10" stroke-linecap="round">
        <line x1="192" y1="208" x2="256" y2="256" />
        <line x1="320" y1="208" x2="256" y2="256" />
        <line x1="192" y1="304" x2="256" y2="256" />
        <line x1="320" y1="304" x2="256" y2="256" />
        <line x1="192" y1="208" x2="320" y2="208" />
        <line x1="192" y1="304" x2="320" y2="304" />
      </g>

      <g fill="#F8FAFC">
        <polygon points="256,220 268,256 256,292 244,256" />
        <polygon points="220,256 256,244 292,256 256,268" />
      </g>

      <g fill="#38BDF8" stroke="#0F172A" stroke-width="6">
        <circle cx="192" cy="208" r="16" />
        <circle cx="320" cy="208" r="16" />
        <circle cx="192" cy="304" r="16" />
        <circle cx="320" cy="304" r="16" />
        <circle cx="256" cy="256" r="18" />
      </g>
    </svg>
  );
}
