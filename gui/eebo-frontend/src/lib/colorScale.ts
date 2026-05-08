// lib/colorScale.ts
import * as d3 from "d3";

export const colorScale = d3.scaleOrdinal<string, string>()
    .range(d3.schemeCategory10);

