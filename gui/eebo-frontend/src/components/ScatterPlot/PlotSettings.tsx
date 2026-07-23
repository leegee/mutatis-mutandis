import { controlsActions } from "../../state/controls.actions";
import { controls } from "../../state/controls.store";

export default function PlotSettings() {
    return <>
        <li class="middle-align top-padding">
            <div class="field middle-align prefix suffix">
                <nav>
                    <div class="slider medium responsive">
                        <input type='range' min={0} max={4} step={0.5}
                            value={controls.bfsOpacity}
                            onInput={(e) => controlsActions.setBfsOpacity(e.currentTarget.value)}
                        />
                        <span><i>brightness_6</i></span>
                    </div>
                </nav>
                <output>BFS Background Opacity</output>
            </div>
        </li>

        <li class="middle-align top-padding">
            <div class="field middle-align prefix suffix">
                <nav>
                    <div class="slider medium responsive">
                        <input type='range' min={0} max={255} step={5}
                            disabled={!controls.showNeighbours}
                            value={controls.neighbourOpacity}
                            onInput={(e) => controlsActions.setNeighbourOpacity(Number(e.currentTarget.value))}
                        />
                        <span><i>brightness_6</i></span>
                    </div>
                </nav>
                <output>Neighbour Opacity</output>
            </div>
        </li>

        <li class="middle-align top-padding">
            <div class="field middle-align prefix suffix">
                <nav>
                    <div class="slider medium responsive">
                        <input type='range' min={1} max={5} step={0.5}
                            value={controls.plotPointScaleFactor}
                            onInput={(e) => controlsActions.setPlotPointScaleFactor(Number(e.currentTarget.value))}
                        />
                        <span><i>lens_blur</i></span>
                    </div>
                </nav>
                <output>Neighbour Opacity</output>
            </div>
        </li>
    </>
}