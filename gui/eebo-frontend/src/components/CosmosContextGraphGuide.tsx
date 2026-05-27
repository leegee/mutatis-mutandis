export default function ConceptGraphGuide() {
  return (
    <section>
      <h1>Comos Graph</h1>
      <h2>User Guide</h2>

      <h3>Events view</h3>
      <p>
        Each <span style="color:rgba(180,255,190,.9)">green</span> node is one
        individual co-occurrence event:  a single sentence or window in a document
        where the concept <strong><code>LIBERTY</code></strong> appeared.
      </p>
      <p>
        <span style="color:rgba(255,220,140,.9)">Amber</span> nodes are the words
        that appeared alongside it in that window. Events that share many neighbours
        will be pulled close together.
      </p>

      <h3>Reading the layout</h3>
      <ul>
        <li>Tightly clustered event nodes = very similar local context (same document, same argument).</li>
        <li>Isolated events = unusual usage:  the concept appeared in a unique context.</li>
        <li>A large amber node with many spokes = a word that co-occurs across many events.</li>
      </ul>

      <h3>Controls</h3>
      <ul>
        <li><strong>Top N</strong>:  how many neighbours each event contributes to the graph.</li>
        <li><strong>Hub spread</strong>:  stretches or compresses the layout.</li>
        <li><strong>Year slider</strong>:  filter to a single year or range.</li>
      </ul>

      <h3>Interactions</h3>
      <ul>
        <li>Hover any node for a tooltip. Click to open the detail panel.</li>
        <li>Drag nodes. Scroll to zoom. Click blank space to deselect.</li>
      </ul>

      <h3>Aggregated view</h3>
      <p>
        Each <span style="color:rgba(180,220,255,.9)">blue</span> node (hub) is a
        distinct token form of <strong><code>LIBERTY</code></strong>:  e.g. singular vs.
        plural, or a specific compound. Size encodes how many co-occurrence events
        that form appears in; brightness encodes how many other hubs it resembles.
      </p>
      <p>
        <span style="color:rgba(255,220,140,.9)">Amber</span> nodes are the words
        most frequently found alongside each hub. The same amber word may appear
        near multiple hubs:  that overlap is what the layout exploits.
      </p>

      <h3>Reading the layout</h3>
      <ul>
        <li>Hubs close together = similar distributional context (they tend to appear near the same words).</li>
        <li>Blue edges between hubs = cosine similarity above the <em>Min sim</em> threshold.</li>
        <li>An amber node between two hubs = that word bridges both usages.</li>
        <li>An amber node hanging off one hub = a word specific to that usage.</li>
      </ul>

      <h3>Controls</h3>
      <ul>
        <li><strong>Max hubs</strong>:  limits how many token forms are shown. Start small.</li>
        <li><strong>Top N</strong>:  how many neighbours each hub contributes to the graph.</li>
        <li><strong>Min sim</strong>:  cosine threshold for drawing hub–hub edges. Lower = more edges.</li>
        <li><strong>Hub spread</strong>:  increases repulsion force to spread hubs apart.</li>
        <li><strong>Year</strong>:  filter to a single year or range to track usage over time.</li>
      </ul>

      <h3>Interactions</h3>
      <ul>
        <li>Hover any node for a tooltip. Click to open the detail panel on the right.</li>
        <li>The detail panel lists top neighbours, document sources, and years.</li>
        <li>Drag nodes. Scroll to zoom. Click blank space to deselect.</li>
      </ul>
    </section>
  );
}
