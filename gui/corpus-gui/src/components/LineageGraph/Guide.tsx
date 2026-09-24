export default function NeighbourhoodBrowserGuide() {
  return (
    <div class="lineage-help">
      <h2>Semantic Lineage Graph</h2>

      <p>
        This graph shows how patterns of meaning associated with a concept change
        through time in the corpus. Rather than tracking a word as if it had
        one fixed meaning, it follows recurring patterns of usage: groups of
        passages whose surrounding language places them in similar semantic
        contexts.
      </p>

      <p>
        Each circle represents a cluster of contextual observations from a
        particular publication period. An observation is an individual occurrence
        of the concept in a text, together with the linguistic context in which it
        appears. A cluster therefore represents a historically situated pattern
        of usage rather than simply a collection of identical words.
      </p>

      <h3>Reading the graph</h3>

      <ul>
        <li>
          <strong>Horizontal position</strong> represents time. Moving from left
          to right follows the development of semantic usage across publication
          years.
        </li>
        <li>
          <strong>Circles</strong> represent semantic clusters. Larger circles
          contain more contextual observations and indicate more frequently
          occurring patterns of usage.
        </li>
        <li>
          <strong>Lines</strong> connect clusters that appear to continue into a
          later period. A line indicates that the later cluster occupies a
          similar region of semantic space to the earlier one.
        </li>
        <li>
          <strong>Multiple outgoing lines</strong> indicate possible branching:
          an earlier pattern of usage appears to divide into several later
          semantic regions.
        </li>
        <li>
          <strong>Multiple incoming lines</strong> indicate possible merging:
          previously distinct patterns of usage converge into a later shared
          semantic region.
        </li>
      </ul>

      <h3>Semantic persistence</h3>

      <p>
        The colour of each circle indicates how closely that cluster remains
        related to the original meaning-pattern from which its lineage began.
        Green indicates stronger persistence; yellow and red indicate increasing
        semantic distance from the founding cluster.
      </p>

      <p>
        This measure helps distinguish gradual continuation from simple chains of
        local similarity. A cluster may be close to its immediate predecessor while
        having moved considerably away from the earlier usage pattern that began
        the lineage.
      </p>

      <h3>Lineages</h3>

      <p>
        A lineage represents a possible trajectory of semantic continuity: a chain
        of related usage patterns across time. A lineage may persist, branch,
        merge with another trajectory, or disappear when no later cluster provides
        a sufficiently similar continuation.
      </p>

      <p>
        These lineages should not be understood as fixed histories of a word's
        meaning. They are computationally identified pathways through a changing
        semantic landscape. They provide starting points for historical
        investigation by showing where patterns of usage appear stable, where they
        divide, and where significant shifts may have occurred.
      </p>

      <h3>Exploring the evidence</h3>

      <p>
        Selecting a cluster reveals examples of the underlying textual evidence:
        individual occurrences from EEBO, their source documents, and nearby
        contextual examples. These allow the you to move from the abstract
        semantic pattern back to the historical texts from which it was derived.
      </p>

      <p>
        The graph is therefore not a model that decides what a concept meant.
        Instead, it is an exploratory map that helps discovery and
        examination of possible trajectories of semantic change.
      </p>
    </div>
  )
}
