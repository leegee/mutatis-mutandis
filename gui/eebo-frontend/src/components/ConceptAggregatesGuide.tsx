export default function Guide() {
    return (
        <div id="ConceptAggregatesGuide">
            <section id="what-is-this" class="surface padding">

                <h2>Concept Aggregates</h2>

                <p>
                    This view summarises the <strong>semantic neighbourhood</strong>
                    of the currently selected concept.
                </p>

                <p>
                    For every occurrence of the concept, the system retrieves its
                    <strong> 25 nearest neighbours</strong> in embedding space using
                    a FAISS similarity search. The returned neighbour events are then
                    aggregated to identify the most frequently associated tokens,
                    documents, and windows.
                </p>

                <h3>Aggregation Process</h3>

                <ol>
                    <li>Identify all events belonging to the selected concept.</li>
                    <li>Retrieve the embedding vector for each event.</li>
                    <li>Run a FAISS nearest-neighbour search (K = 25).</li>
                    <li>Collect the returned neighbour events.</li>
                    <li>Count how often neighbour tokens appear.</li>
                    <li>Count how often neighbour documents appear.</li>
                    <li>Rank the results by frequency.</li>
                </ol>

                <p>
                    The aggregates displayed here are pre-computed and stored in
                    the <code>concept_aggregate</code> table.
                </p>

                <h3>Top Tokens</h3>

                <p>
                    The Top Tokens table shows which tokens most frequently occur
                    among the retrieved nearest-neighbour events.
                </p>

                <table class="stripes">
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Rank</td>
                            <td>Position within the frequency ranking.</td>
                        </tr>
                        <tr>
                            <td>Token</td>
                            <td>
                                The neighbour token returned by the similarity search.
                            </td>
                        </tr>
                        <tr>
                            <td>Count</td>
                            <td>
                                Number of times the token appeared among all retrieved
                                neighbour events.
                            </td>
                        </tr>
                    </tbody>
                </table>

                <p>
                    A high count indicates a strong semantic association with the
                    selected concept.
                </p>

                <h3>Top Documents</h3>

                <p>
                    The Top Documents table shows which documents contribute the
                    largest number of neighbour events.
                </p>

                <table class="stripes">
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Rank</td>
                            <td>Position within the document ranking.</td>
                        </tr>
                        <tr>
                            <td>Document ID</td>
                            <td>
                                Identifier of the document contributing neighbour events.
                            </td>
                        </tr>
                        <tr>
                            <td>Count</td>
                            <td>
                                Total number of neighbour events originating from the document.
                            </td>
                        </tr>
                        <tr>
                            <td>Avg per Event</td>
                            <td>
                                Average neighbour matches contributed by the document
                                per concept event.
                            </td>
                        </tr>
                        <tr>
                            <td>% of Neighbours</td>
                            <td>
                                Percentage of all retrieved neighbour positions
                                represented by this document.
                            </td>
                        </tr>
                    </tbody>
                </table>

                <h3>Metric Definitions</h3>

                <h4>Number of Events</h4>

                <p>
                    Each concept has an associated event count
                    (<code>nEvents</code>) representing the total number of
                    occurrences analysed for that concept.
                </p>

                <pre>
                    nEvents = number of concept events
                </pre>

                <h4>Average per Event</h4>

                <pre>
                    Avg per Event = Count ÷ nEvents
                </pre>

                <p>
                    This indicates how frequently a document appears among the
                    neighbours of an average concept occurrence.
                </p>

                <h4>Percentage of Neighbours</h4>

                <p>
                    Each concept event retrieves up to 25 nearest neighbours.
                </p>

                <pre>
                    Total Neighbour Slots = nEvents × 25

                    % of Neighbours =
                    (Count ÷ Total Neighbour Slots) × 100
                </pre>

                <p>
                    This metric shows how much of the overall neighbour space is
                    occupied by a particular document.
                </p>

                <h3>Interpreting Results</h3>

                <ul>
                    <li>
                        Frequently occurring tokens often represent concepts,
                        themes, or vocabulary closely related to the selected concept.
                    </li>
                    <li>
                        Frequently occurring documents may contain concentrated
                        discussions of the concept or related topics.
                    </li>
                    <li>
                        Higher Avg per Event values indicate stronger and more
                        consistent associations across concept occurrences.
                    </li>
                    <li>
                        Higher % of Neighbours values indicate a larger share of
                        the semantic neighbourhood is contributed by a document.
                    </li>
                </ul>

                <h3>Important Note</h3>

                <p>
                    These aggregates are derived from <strong>embedding-space
                        similarity</strong>, not from a fixed textual context window.
                    A token or document appears in these rankings because its
                    events were retrieved as semantic neighbours during the FAISS
                    search process.
                </p>

            </section>
        </div>
    );
}