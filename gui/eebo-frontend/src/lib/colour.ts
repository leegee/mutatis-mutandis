// Deterministic hash so the same value always maps to the same colour,
// regardless of dataset load order or concept switching.
function hashString(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

// Generates a perceptually spread palette using golden-ratio hue stepping.
// Returns [r, g, b, a] arrays suitable for deck.gl.
export function buildColorMap(
  values: string[],
  topN = 24
): Map<string, [number, number, number, number]> {
  const counts = new Map<string, number>();

  for (const v of values) {
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }

  const sorted = [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([v]) => v);

  const map = new Map<string, [number, number, number, number]>();
  const golden = 0.6180339887;

  sorted.forEach((v, i) => {
    const hue = ((hashString(v) / 0xffffffff + i * golden) % 1) * 360;

    map.set(
      v,
      hslToRgb(hue, 0.72, 0.62)
    );
  });

  return map;
}

export function hslToRgb(
  h: number,
  s: number,
  l: number
): [number, number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;

  let r = 0;
  let g = 0;
  let b = 0;

  if (h < 60) {
    r = c;
    g = x;
  } else if (h < 120) {
    r = x;
    g = c;
  } else if (h < 180) {
    g = c;
    b = x;
  } else if (h < 240) {
    g = x;
    b = c;
  } else if (h < 300) {
    r = x;
    b = c;
  } else {
    r = c;
    b = x;
  }

  return [
    Math.round((r + m) * 255),
    Math.round((g + m) * 255),
    Math.round((b + m) * 255),
    220,
  ];
}

export type CssClusterColor = {
  bg: string;
  fg: string;
};

export function buildCssColorMap(
  values: string[],
  topN = 24
): Map<string, CssClusterColor> {
  const rgbMap = buildColorMap(values, topN);

  const cssMap = new Map<string, CssClusterColor>();

  for (const [key, [r, g, b, a]] of rgbMap) {
    cssMap.set(key, {
      bg: `rgba(${ r }, ${ g }, ${ b }, ${ (a / 255).toFixed(2) })`,
      fg: pickForeground(r, g, b),
    });
  }

  return cssMap;
}

export function pickForeground(
  r: number,
  g: number,
  b: number
): string {
  const channel = (v: number) => {
    v /= 255;

    return v <= 0.03928
      ? v / 12.92
      : Math.pow((v + 0.055) / 1.055, 2.4);
  };

  const luminance =
    0.2126 * channel(r) +
    0.7152 * channel(g) +
    0.0722 * channel(b);

  return luminance > 0.179
    ? "#000"
    : "#fff";
}
