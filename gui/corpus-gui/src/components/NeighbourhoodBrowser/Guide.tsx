export default function NeighbourhoodBrowserGuide() {
    return (
        <div class="#NeighbourhoodBrowserGuide">
            <p>A guide to finding and exploring word associations across documents.</p>

            <nav class="scroll">
                <ol>
                    <a href="#what-is-this">What is this tool?</a>
                    <a href="#layout">The three panels</a>
                    <a href="#events">Working with events</a>
                    <a href="#tokens">Exploring neighbour tokens</a>
                    <a href="#documents">Reading documents</a>
                    <a href="#filtering">Filtering by year</a>
                    <a href="#footer">The status bar</a>
                    <a href="#keyboard">Keyboard shortcuts</a>
                </ol>
            </nav>

            <section id="what-is-this">
                <h3>What is this tool?</h3>
                <p>
                    The Neighbourhood Browser shows you which words appear near a chosen concept
                    across a collection of documents. Think of it as a way to ask:
                    <em>"When this idea shows up in the texts, what other words tend to show up alongside it?"</em>
                </p>
                <p>
                    Those nearby words are called <strong>neighbour tokens</strong>. Each time a
                    concept appears in a document it creates an <strong>event</strong>. The browser
                    lets you explore all of those events at once, or zoom into a single one.
                </p>
            </section>


            <section id="layout">
                <h3>The three panels</h3>
                <p>The screen is divided into three columns that work together:</p>

                <div class="row top-align">
                    <div style="width:33%">
                        <div class="italic">Left</div>
                        <div class="col-desc"><strong>Events</strong><br />Every occurrence of the concept, one row per document.</div>
                    </div>
                    <div style="width:33%">
                        <div class="italic">Centre</div>
                        <div class="col-desc"><strong>Neighbour tokens</strong><br />Words that appear near the concept in the selected event (or across all events).</div>
                    </div>
                    <div style="width:33%">
                        <div class="italic">Right</div>
                        <div class="col-desc"><strong>Documents</strong><br />The source texts associated with the selected token or event.</div>
                    </div>
                </div>

                <p>
                    Above the neighbour tokens, a short passage of text appears whenever an event
                    or document is active. This shows you the concept in its original context.
                </p>
            </section>


            <section id="events">
                <h3>Working with events</h3>
                <p>
                    Each row in the left panel is one event &mdash; one place in one document where the
                    concept occurs. The row shows the document's year and its ID.
                </p>

                <h4>Selecting an event</h4>
                <p>
                    Click any event row to select it. The centre panel will update to show only
                    the neighbour tokens for that event, and the text excerpt at the top of the
                    centre panel will show you the surrounding passage.
                </p>
                <p>
                    Click the same row again to deselect it and return to the full picture.
                </p>

                <div class="tip">
                    <strong>Tip</strong>
                    Rows that are dimmed are not linked to the word you have focused in the centre
                    panel. This is normal &mdash; it just means those events don't contain that word as
                    a neighbour.
                </div>
            </section>


            <section id="tokens">
                <h3>Exploring neighbour tokens</h3>
                <p>
                    The centre panel shows the words that appear alongside your concept. How they
                    are displayed depends on whether you have an event selected.
                </p>

                <h4>No event selected &mdash; the word cloud</h4>
                <p>
                    When nothing is selected, all neighbour tokens from across every event are
                    shown as clickable chips. Larger chips mean that word appeared as a neighbour
                    in more events. If you are browsing a date range, a small sparkline chart on
                    each chip shows how common that word was in each year.
                </p>

                <h4>Event selected &mdash; the scored list</h4>
                <p>
                    When an event is selected, you see a ranked list of neighbour tokens specific
                    to that event. Each row has a short bar on the left &mdash; a longer bar means a
                    higher similarity score, i.e. that word was more closely associated with the
                    concept in this particular occurrence.
                </p>

                <h4>Focusing on a token</h4>
                <p>
                    Click any token (in either view) to <strong>focus</strong> it. Focusing does
                    two things:
                </p>
                <ol class="steps">
                    <li>The left panel dims any events that do <em>not</em> contain that token as a neighbour, so you can see at a glance how widespread the association is.</li>
                    <li>The right panel updates to list every document in which that token appears near the concept.</li>
                </ol>
                <p>Click the token again to clear the focus.</p>
            </section>


            <section id="documents">
                <h3>Reading documents</h3>
                <p>
                    The right panel lists documents. Which documents appear there depends on what
                    you have selected:
                </p>
                <ol class="steps">
                    <li><strong>Nothing selected:</strong> the panel is empty and prompts you to pick an event or token.</li>
                    <li><strong>An event selected:</strong> the single source document for that event appears.</li>
                    <li><strong>A token focused:</strong> every document that contains both the concept and that token appears, sorted by year.</li>
                </ol>

                <h4>Previewing a document</h4>
                <p>
                    Click a document row to load a text excerpt at the top of the centre panel.
                    This excerpt is centred on the relevant part of the document &mdash; the passage
                    where the concept (and the focused token, if any) appears.
                </p>
                <p>
                    Click the document button below the excerpt to open the full document.
                </p>
                <p>
                    Click the same document row again to close the preview.
                </p>

                <div class="tip">
                    <strong>Tip</strong>
                    You can switch between documents in the right panel without losing your focused
                    token. Each click just swaps the excerpt shown in the centre.
                </div>
            </section>


            <section id="filtering">
                <h3> Filtering by year</h3>
                <p>
                    At the top of the page, the controls bar lets you set a date range. Only
                    events from documents published within that range will be shown. The status
                    bar at the bottom of the page shows your current range if it differs from the
                    full span of the data.
                </p>
                <p>
                    When a date range is active, the sparkline charts on each token chip show how
                    that word's usage rises and falls across the selected years.
                </p>
            </section>


            <section id="footer">
                <h3>The status bar</h3>
                <p>
                    The bar along the very bottom of the screen gives you a quick summary of what
                    is currently loaded and active:
                </p>
                <ol class="steps">
                    <li><strong>Events</strong> &mdash; how many events are visible with the current filters.</li>
                    <li><strong>Event-linked tokens</strong> &mdash; how many distinct neighbour words appear across those events.</li>
                    <li><strong>Documents</strong> &mdash; how many documents are listed in the right panel right now.</li>
                    <li><strong>Focus</strong> &mdash; if a token is focused, its name and counts are shown here.</li>
                    <li><strong>Year range</strong> &mdash; shown only when you have narrowed the date filter.</li>
                </ol>
            </section>


            <section id="keyboard">
                <h3>Keyboard shortcuts</h3>
                <p>
                    Once you have selected an event, you can move up and down the event list
                    without using the mouse:
                </p>
                <p>
                    <kbd>↑</kbd> or <kbd>←</kbd>:  go to the previous event<br />
                    <kbd>↓</kbd> or <kbd>→</kbd>:  go to the next event
                </p>
            </section>

        </div>
    )
};
