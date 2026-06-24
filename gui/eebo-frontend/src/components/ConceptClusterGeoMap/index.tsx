import { createResource } from 'solid-js';
import GeoMap from './GeoMap';
import { controls } from '../../state/controls.store';
import { loadDatasets } from '../ScatterPlot/loadScatterDatasets';


export default function ConceptClusterGeoMap() {

    const sharedKey = () => ({
        concepts: controls.conceptSelection,
        fromYear: controls.fromYear,
        toYear: controls.toYear,
        yearMode: controls.yearMode,
    });

    const [conceptDatasets, { refetch }] = createResource(
        () => ({ ...sharedKey(), dataType: "concept" }),
        loadDatasets
    );

    return (<GeoMap clusterDatasets={conceptDatasets()} />)
}
