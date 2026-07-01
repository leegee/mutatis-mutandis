
export default function DiachronicChartGuide() {
  return (
    <section id="ScatterPlotGuide">
      <p>
        The scatter plot shows embedding relationships between concepts
        through UMP and PACMAP collapsisings of 768 dimensions of real numbers
        to integers in a two-dimensional space.
      </p>
      <p>
        Select a point by clicking, add another point by <kbd>SHIFT</kbd>-clicking. Hold <kbd>CTRL</kbd> whilst  clicking and dragging to select multiple points.
        Once you have thus raised summary reports of the selected mini-corpus, you may wish to download your selection,
        or copy it to your clipboard. Several styles of export are available from the sidebar.
      </p>
      <p>
        Other options are exploratory prototypes - send parts of selections to <code>llama-3.3-70b-versatile</code>
        to see its attempt at classification which really needs luca to unlock larger payloads that can carry multiople clusters,
        as a differential task might yield better results than one where there is no supplied refereant but where the frame of
        reference is solely in the prompts.
      </p>
      <p>
        <em>TODO</em> &mdash; let's add a new layer with opacity+visibility controls, to hold results GPT definitions of clusters - anchor at centroid of same.
      </p>
    </section>
  )
}
