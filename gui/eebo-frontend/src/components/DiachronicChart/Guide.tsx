export default function DiachronicChartGuide() {
    return (
        <div class="#DiachronicChartGuide">
            <p>
                A guide to understanding the diachronic token flow visualization — a time-based
                chart showing how words appear, persist, and disappear across years.
            </p>

            <nav class="scroll">
                <ol>
                    <a href="#what-is-this">What is this visualization?</a>
                    <a href="#tokens">Token cells and meaning</a>
                    <a href="#time">Birth, death, and continuation</a>
                    <a href="#links">Connections across years</a>
                    <a href="#focus">Focusing on a token</a>
                    <a href="#controls">Controls and settings</a>
                    <a href="#data">How the data is structured</a>
                    <a href="#interpretation">How to read the chart</a>
                </ol>
            </nav>

            <section id="what-is-this">
                <h3>What is this visualization?</h3>
                <p>
                    This chart shows how tokens (words or concepts) evolve over time. Each column
                    represents a year, and each row shows the most important tokens for that year.
                </p>
                <p>
                    The goal is to make it easy to see:
                    <em>
                        {" "}which words appear, which disappear, and which persist across time.
                    </em>
                </p>
                <p>
                    When a token appears in multiple years, it is connected with a line, forming
                    a visual trajectory through time.
                </p>
            </section>

            <section id="tokens">
                <h3>Token cells and meaning</h3>
                <p>
                    Every rectangle in the chart represents a token appearing in a specific year.
                    The label shows the token itself.
                </p>

                <h4>Visual encoding</h4>
                <ol class="steps">
                    <li><strong>Position:</strong> row position indicates rank (higher = more important).</li>
                    <li><strong>Column:</strong> indicates year.</li>
                    <li><strong>Opacity:</strong> fades out unrelated tokens when a focus is active.</li>
                    <li><strong>Stroke:</strong> highlights the currently focused token.</li>
                </ol>

                <p>
                    Tokens are also colored according to how they behave over time.
                </p>
            </section>

            <section id="time">
                <h3>Birth, death, and continuation</h3>
                <p>
                    Each token occurrence is classified based on whether it appears before or after
                    the current year.
                </p>

                <ol class="steps">
                    <li>
                        <strong>Birth</strong> — the token appears for the first time in the dataset.
                    </li>
                    <li>
                        <strong>Death</strong> — the token appears for the last time in the dataset.
                    </li>
                    <li>
                        <strong>Birth-death</strong> — the token appears only in a single year.
                    </li>
                    <li>
                        <strong>Continuation</strong> — the token exists both before and after this year.
                    </li>
                </ol>

                <p>
                    These states are used to assign color, helping you quickly see how stable or
                    transient a token is across time.
                </p>
            </section>

            <section id="links">
                <h3>Connections across years</h3>
                <p>
                    When a token appears in multiple years, it is connected by a curved line between
                    columns.
                </p>

                <h4>What the lines mean</h4>
                <ol class="steps">
                    <li>
                        Lines connect identical tokens across consecutive years.
                    </li>
                    <li>
                        The curve shows movement through time, even when rank changes.
                    </li>
                    <li>
                        Line opacity indicates whether a token is currently focused.
                    </li>
                </ol>

                <p>
                    These links turn isolated yearly rankings into continuous temporal trajectories.
                </p>
            </section>

            <section id="focus">
                <h3>Focusing on a token</h3>
                <p>
                    Clicking a token activates focus mode. This isolates its trajectory across time.
                </p>

                <h4>What happens in focus mode</h4>
                <ol class="steps">
                    <li>Only the selected token remains visually prominent.</li>
                    <li>All unrelated tokens are faded.</li>
                    <li>Its connections across years are highlighted.</li>
                    <li>Gaps in appearance are bridged visually where possible.</li>
                </ol>

                <p>
                    Clicking the token again clears focus and restores the full view.
                </p>
            </section>

            <section id="controls">
                <h3>Controls and settings</h3>
                <p>
                    The control bar above the chart affects how data is aggregated and displayed.
                </p>

                <h4>Sorting</h4>
                <p>
                    Tokens can be ranked either by frequency or cosine similarity score.
                </p>

                <h4>Smoothing</h4>
                <p>
                    A smoothing slider controls how many adjacent years are blended together.
                    Higher smoothing reduces noise and produces broader trends.
                </p>

                <p>
                    When smoothing is active, year labels become ranges instead of single values.
                </p>
            </section>

            <section id="data">
                <h3>How the data is structured</h3>
                <p>
                    The visualization is built from yearly slices of token data.
                </p>

                <ol class="steps">
                    <li>
                        Each year contains a ranked list of tokens.
                    </li>
                    <li>
                        Each token includes frequency and scoring metadata.
                    </li>
                    <li>
                        A map structure groups tokens by year for fast lookup.
                    </li>
                </ol>

                <p>
                    Internally, the system computes both raw and smoothed versions of these slices
                    to support different views.
                </p>
            </section>

            <section id="interpretation">
                <h3>How to read the chart</h3>
                <p>
                    The best way to interpret the visualization is to follow individual tokens across
                    time.
                </p>

                <p>
                    Look for:
                </p>

                <ol class="steps">
                    <li>Long horizontal trajectories → stable, persistent concepts.</li>
                    <li>Short-lived appearances → emerging or fading ideas.</li>
                    <li>Rank shifts → changing importance over time.</li>
                    <li>Dense link clusters → periods of conceptual stability.</li>
                </ol>

                <p>
                    Together, these patterns show how a vocabulary evolves across the dataset.
                </p>
            </section>
        </div>
    );
}