#!/bin/bash
TYPE=svg
OUTFILE_TYPE="${1:-../output/pamphlet_pipeline_output.$TYPE}"
OUTFILE_DOT="${1:-../output/pamphlet_pipeline_output.dot}"


# https://graphviz.org/doc/info/colors.html#brewer
export COLORSCHEME="gnbu5"

# Set hierarchical colours (1 = lightest/top-level, 2 = subcluster, 3 = nodes)
export PAGE_FG="#666677"
export PAGE_BG="#DDDDE0"

export HELP_FG="black"
export HELP_BG="#EECCCC"

export CLUSTER_FG="5"
export CLUSTER_BG="1"

export SUBCLUSTER_FG="5"
export SUBCLUSTER_BG="2"

export NODE_FG="black"
export NODE_BG="3"

export OUTPUT_FG="white"
export OUTPUT_BG="4"
export OUTPUT_FONTSIZE="32"
export OUTPUT_FONTWEIGHT="bold"
export OUTPUT_FONTNAME="Figtree"

export FONT_NODE="24"
export FONT_H1="48"
export FONT_H2="32"
export FONT_H3="24"

# Read template and substitute variables
sed \
  -e "s/\${COLORSCHEME}/$COLORSCHEME/g" \
  -e "s/\${OUTPUT_FG}/$OUTPUT_FG/g" \
  -e "s/\${OUTPUT_BG}/$OUTPUT_BG/g" \
  -e "s/\${OUTPUT_FONTSIZE}/$OUTPUT_FONTSIZE/g" \
  -e "s/\${OUTPUT_FONTWEIGHT}/$OUTPUT_FONTWEIGHT/g" \
  -e "s/\${OUTPUT_FONTNAME}/$OUTPUT_FONTNAME/g" \
  -e "s/\${HELP_FG}/$HELP_FG/g" \
  -e "s/\${HELP_BG}/$HELP_BG/g" \
  -e "s/\${PAGE_FG}/$PAGE_FG/g" \
  -e "s/\${PAGE_BG}/$PAGE_BG/g" \
  -e "s/\${CLUSTER_FG}/$CLUSTER_FG/g" \
  -e "s/\${CLUSTER_BG}/$CLUSTER_BG/g" \
  -e "s/\${SUBCLUSTER_FG}/$SUBCLUSTER_FG/g" \
  -e "s/\${SUBCLUSTER_BG}/$SUBCLUSTER_BG/g" \
  -e "s/\${NODE_FG}/$NODE_FG/g" \
  -e "s/\${NODE_BG}/$NODE_BG/g" \
  -e "s/\${FONT_NODE}/$FONT_NODE/g" \
  -e "s/\${FONT_H1}/$FONT_H1/g" \
  -e "s/\${FONT_H2}/$FONT_H2/g" \
  -e "s/\${FONT_H3}/$FONT_H3/g" \
  pamphlet_pipeline.dot.template > "$OUTFILE_DOT"

dot -T${TYPE} "$OUTFILE_DOT" -o "$OUTFILE_TYPE"

# cat "$OUTFILE_DOT"

echo "Dot generated at $OUTFILE_DOT"
echo "$TYPE graph generated at $OUTFILE_TYPE"
