import type { ProjectionModeType } from "../../state/controls.store";
import type { PointData } from "./types";

export function fitDatasetBounds(
  points: PointData[],
  projection: ProjectionModeType
): { cx: number, cy: number, extent: number } | undefined {
  if (!points.length) return;

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  for (const p of points) {
    const [x, y] = projection === "global"
      ? [p.gnx, p.gny]
      : [p.nx, p.ny];

    if (x == null || y == null) continue;

    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }

  if (!Number.isFinite(minX)) return;

  const width = maxX - minX;
  const height = maxY - minY;

  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;

  // Orthographic zoom is logarithmic:
  // larger dataset extent => lower zoom
  const extent = Math.max(width, height);

  return { extent, cx, cy };
}
