// GlowScatterplotLayer.tsx

import { CompositeLayer } from "@deck.gl/core";
import { ScatterplotLayer } from "@deck.gl/layers";
import type { ScatterplotLayerProps } from "@deck.gl/layers";


type GlowScatterplotLayerProps<DataT extends {}> =
  ScatterplotLayerProps<DataT> & {
    glowScale?: number;
    glowAlpha?: number;
  };


// Internal visual-only layer
class GlowScatterLayer<DataT extends {} = any>
  extends ScatterplotLayer<DataT> {
  getShaders() {
    const shaders = super.getShaders();

    shaders.inject = {
      "fs:#main-end": `
        float d = length(unitPosition);

        // soft halo
        float glow = 1.0 - smoothstep(0.15, 1.0, d);

        // fade edge
        fragColor.a *= glow * 0.65;

        // slight bloom
        fragColor.rgb += fragColor.rgb * glow * 0.35;
      `
    };

    return shaders;
  }
}



export class GlowScatterplotLayer<DataT extends {} = any>
  extends CompositeLayer<GlowScatterplotLayerProps<DataT>> {
  static layerName = "GlowScatterplotLayer";

  renderLayers() {
    const {
      id,
      data,
      glowScale = 10,
      glowAlpha = 40,
      getPosition,
      getFillColor,
      getRadius,
      coordinateSystem,
      radiusUnits = "pixels",
      opacity = 1,
      ...coreProps
    } = this.props;


    const radiusAccessor =
      typeof getRadius === "function"
        ? getRadius
        : () => getRadius ?? 1;


    const colorAccessor =
      typeof getFillColor === "function"
        ? getFillColor
        : () => getFillColor ?? [255, 255, 255, 255];


    return [
      // Visual glow
      new GlowScatterLayer<DataT>({
        id: `${ id }-glow`,
        data,
        coordinateSystem,
        getPosition,
        getRadius: (d: DataT, info: any) => radiusAccessor(d, info) * glowScale,
        getFillColor: (d: DataT, info: any) => {
          const c = colorAccessor(d, info);
          return [
            c[0],
            c[1],
            c[2],
            glowAlpha,
          ];
        },
        radiusUnits,
        opacity,
        filled: true,
        stroked: false,
        pickable: false,
      }),


      // Interactive core
      new ScatterplotLayer<DataT>({
        id: `${ id }-core`,
        data,
        coordinateSystem,
        getPosition,
        getRadius: radiusAccessor,
        getFillColor: colorAccessor,
        radiusUnits,
        opacity,
        ...coreProps,
      }),
    ];
  }
}