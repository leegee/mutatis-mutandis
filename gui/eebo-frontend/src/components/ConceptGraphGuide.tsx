export default function ConceptGraphGuide() {
  return (
    <article>
      <h1>Concept Graph — User Guide</h1>

      <h2>What you are looking at</h2>

      <p>
        Each graph shows the <strong>semantic neighbourhood structure</strong>
        of a concept as it appears across the EEBO corpus.
        It is not a map of words that <em>mean</em> the same thing as the concept —
        it is a map of words that <em>appear in similar contexts</em> to it,
        across thousands of individual token occurrences in early modern pamphlet texts.
      </p>

      <p>
        The underlying principle is Firthian:
        a word is known by the company it keeps.
        Every time a concept such as <em>liberty</em> or <em>conscience</em>
        appears in the corpus, the model records which other words appear
        in its immediate semantic neighbourhood —
        the contextual environment the transformer model perceives
        around that occurrence.
      </p>

      <p>
        The graph aggregates those neighbourhood relationships
        across all occurrences,
        surfacing the relational structure of the concept's usage
        across time, genre, and discourse context.
      </p>


      <h2>Nodes</h2>

      <p>
        Each node is a word that appeared frequently
        in the semantic neighbourhoods of the selected concept.
      </p>

      <p>
        <strong>Size</strong> encodes degree —
        how many other words in the graph this word co-occurs with.
        Larger nodes are more relationally central:
        they appear alongside many other words in the concept's context,
        not just frequently.
      </p>

      <p>
        <strong>Brightness</strong> also encodes degree.
        Brighter, more blue-white nodes are hubs;
        darker, more slate-coloured nodes are peripheral.
        A large, bright node is a word that sits at the centre
        of the concept's semantic world.
      </p>

      <h2>Edges</h2>

      <p>
        Each edge connects two words that appeared together
        in the neighbourhood of the same concept occurrence —
        they were both close to the concept at the same moment
        in the same text.
      </p>

      <p>
        <strong>Thickness</strong> encodes how often this pair
        co-occurred in neighbourhood space.
        A thick edge means the two words were repeatedly found together
        around the concept, across many documents.
      </p>

      <p>
        <strong>Opacity</strong> also encodes co-occurrence frequency.
        Faint edges represent occasional pairings;
        solid edges represent persistent ones.
      </p>

      <p>
        <strong>Colour</strong> runs as a gradient
        from one node's colour to the other's.
        This means edges between two hub nodes appear bright blue-white;
        edges between a hub and a peripheral node fade from bright to dark;
        edges between two peripheral nodes are dark throughout.
      </p>

      <p>
        The overall effect is that the densely connected core of the graph glows,
        while the periphery recedes.
      </p>


      <h2>Controls</h2>

      <p>
        <strong>Concept</strong> selects which concept's neighbourhood structure
        to display.
        Each concept is built from a set of word forms drawn from the corpus,
        including orthographic variants common in early modern English.
      </p>

      <p>
        <strong>Max nodes</strong> sets a ceiling on how many words are shown.
        The graph always displays the highest-degree nodes first —
        the words most centrally connected in neighbourhood space —
        and drops lower-degree words to meet the limit.
      </p>

      <p>
        Reducing this number focuses the view on the semantic core;
        increasing it reveals more of the periphery.
      </p>

      <p>
        <strong>Min edge</strong> sets the minimum number
        of co-occurrences required for an edge to appear.
        At low values the graph is dense and shows occasional associations;
        at higher values only persistent, recurring relationships survive.
      </p>

      <p>
        Raising this threshold is useful for identifying
        the stable semantic companions of a concept —
        the words that appear with it not once or twice
        but consistently across the corpus.
      </p>

      <p>
        The node and edge counts shown to the right of the controls
        reflect the graph <em>after</em> both filters are applied.
      </p>


      <h2>Navigation</h2>

      <p>
        <strong>Scroll</strong> to zoom in and out.
        At high zoom levels individual node labels become readable
        for dense regions of the graph.
      </p>

      <p>
        <strong>Drag the background</strong> to pan across the graph.
      </p>

      <p>
        <strong>Drag individual nodes</strong> to reposition them.
        The simulation will continue running around the moved node;
        releasing it returns it to the simulation.
      </p>

      <p>
        <strong>Hover over a node</strong> to see its label and degree
        in a tooltip.
      </p>


      <h2>What to look for</h2>

      <p>
        <strong>Tight clusters</strong> —
        groups of densely interconnected nodes —
        suggest semantic sub-fields:
        words that consistently travel together around the concept,
        implying a coherent domain of usage.
      </p>

      <p>
        In a concept like <em>liberty</em>,
        one cluster might be legal and constitutional,
        another theological,
        another polemical.
      </p>

      <p>
        <strong>Bridge nodes</strong> —
        words that connect otherwise separate clusters —
        are particularly significant.
        They suggest semantic overlap or contested ground
        between two domains of usage.
      </p>

      <p>
        <strong>Isolated or peripheral nodes</strong> —
        connected by thin, faint edges —
        represent occasional or idiosyncratic associations,
        possibly tied to specific documents or authors
        rather than general usage patterns.
      </p>

      <p>
        <strong>Absence</strong> is also meaningful.
        If a word you expect to find near a concept does not appear,
        it may mean the association is present in the texts
        but below the co-occurrence threshold,
        or that the relationship is more diffuse than intuition suggests.
      </p>


      <h2>Relationship to the underlying data</h2>

      <p>
        The graph is a projection over the full semantic event ledger —
        a visualisation of aggregate neighbourhood structure,
        not the primary data.
      </p>

      <p>
        Every node and edge is traceable back
        to specific token occurrences
        in specific EEBO documents.
        The graph summarises;
        the texts ground it.
      </p>

      <p>
        Changing the filters does not change the underlying data —
        it changes which part of the neighbourhood structure is visible.
        The same corpus events underlie all views.
      </p>

    </article>
  )
}