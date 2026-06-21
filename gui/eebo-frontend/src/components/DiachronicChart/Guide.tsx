import { C_BIRTH, C_BIRTH_DEATH, C_CONTINUATION, C_DEATH } from "./DiachronicChart";

export default function DiachronicChartGuide() {
    return (
        <div class="#DiachronicChartGuide">
            <section id="what-is-this" class="surface padding">
                <h5>What is this visualization?</h5>
                <p>
                    Each column
                    represents a year, and each row shows the tokens ordered by either
                    frequency or cosine similarity.
                </p>
                <p>
                    When a token appears in multiple years, it is connected with a line, forming
                    a visual trajectory through time. This can be highlighted by
                    <a href="#selecting">selecting</a>.
                    that term.
                </p>
            </section>

            <section id="time" class="surface padding">
                <h5>Birth, death, and continuation</h5>
                <p>
                    Each token occurrence is classified based on whether it appears before or after
                    the current year.
                </p>

                <ul class="list border no-space">
                    <li>
                        <strong style={`color: ${ C_BIRTH }`}>Birth</strong> the token appears for the first time in the dataset.
                    </li>
                    <li>
                        <strong style={`color: ${ C_DEATH }`}>Death</strong> the token appears for the last time in the dataset.
                    </li>
                    <li>
                        <strong style={`color: ${ C_BIRTH_DEATH }`}>Birth-death</strong> the token appears only in a single year.
                    </li>
                    <li>
                        <strong style={`color: ${ C_CONTINUATION }`}>Continuation</strong> the token exists both before and after this year.
                    </li>
                </ul>
            </section>

            <section id="controls" class="surface padding">
                <h5 id="selecting">Controls and settings</h5>
                <p>
                    Navigate the map with the mouse, or by touch, or using the keyboard cursor keys,
                    the <kbd>Escape</kbd> key,
                    and the <kbd>Enter</kbd> key with modifiers: <kbd>SHIFT</kbd> to change direction,
                    any other modifier to jump to the end of the chain.
                </p>
                <p>
                    The control bar above the chart affects how data is aggregated and displayed.
                </p>

                <h6>Sorting</h6>
                <p>
                    Tokens can be ranked either by frequency or cosine similarity score.
                </p>

                <h6>Smoothing</h6>
                <p>
                    A smoothing slider controls how many adjacent years are blended together.
                    Higher smoothing reduces noise and produces broader trends.
                </p>

                <p>
                    When smoothing is active, year labels become ranges instead of single values.
                </p>
            </section>

            <section id="interpretation" class="surface padding">
                <h5>How to read the chart</h5>
                <p>
                    The best way to interpret the visualization is to follow individual tokens across
                    time.
                </p>

                <p>
                    Look for:
                </p>

                <ul class="list border no-space">
                    <li>Long horizontal trajectories → stable, persistent concepts.</li>
                    <li>Short-lived appearances → emerging or fading ideas.</li>
                    <li>Rank shifts → changing importance over time.</li>
                    <li>Dense link clusters → periods of conceptual stability.</li>
                </ul>

                <p>
                    Together, these patterns show how a vocabulary evolves across the dataset.
                </p>
            </section>
        </div>
    );
}