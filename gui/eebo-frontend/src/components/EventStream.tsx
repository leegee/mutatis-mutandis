// src/components/EventStream.tsx

import {
    selectedConcept,
    selectedSlice,
    setSelectedConcept,
    setSelectedSlice,
    setSelectedEventId
} from "../state/selection";

import "./EventStream.css";

type Props = {
    concepts: string[];
    slicesByConcept: Record<
        string,
        string[]
    >;
};

export default function EventStream(
    props: Props
) {

    return (
        <article class="stream">
            {props.concepts.map(concept => {

                const activeConcept = selectedConcept() === concept;

                return (
                    <ul class="stream-concept list border no-space right-margin max">
                        <li class="max"
                            onClick={() => {
                                setSelectedConcept(concept);
                                setSelectedSlice(null);
                                setSelectedEventId(null);
                            }}
                        >
                            {concept}
                        </li>

                        {activeConcept && (
                            <li class="no-space max right-align">
                                <ul class="stream-slices list border no-space max right-align">
                                    {(props.slicesByConcept[concept] ?? [])
                                        .sort()
                                        .map(slice => {

                                            const activeSlice = selectedSlice() === slice;
                                            return (
                                                <li class={activeSlice ? 'underline bold' : ''}
                                                    onClick={() => {
                                                        setSelectedSlice(slice);
                                                        setSelectedEventId(null);
                                                    }}
                                                >
                                                    {slice}
                                                </li>
                                            );
                                        })}
                                </ul>
                            </li>
                        )}
                    </ul>
                );
            })}

        </article>
    );
}
