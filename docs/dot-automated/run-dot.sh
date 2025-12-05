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

export CLUSTER_FONTNAME="Figtree ExtraBold"
export CLUSTER_FONTSIZE="48"
export CLUSTER_FG="5"
export CLUSTER_BG="1"

export SUBCLUSTER_FONTNAME="Figtree SemiBold"
export SUBCLUSTER_FONTSIZE="48"
export SUBCLUSTER_FG="5"
export SUBCLUSTER_BG="2"

export NODE_FONTSIZE="24"
export NODE_FG="black"
export NODE_BG="3"

export OUTPUT_FONTSIZE="32"
export OUTPUT_FONTNAME="Figtree"
export OUTPUT_FG="white"
export OUTPUT_BG="4"


# Read template and substitute variables
sed \
  -e "s/\${COLORSCHEME}/$COLORSCHEME/g" \
  -e "s/\${OUTPUT_FG}/$OUTPUT_FG/g" \
  -e "s/\${OUTPUT_BG}/$OUTPUT_BG/g" \
  -e "s/\${OUTPUT_FONTSIZE}/$OUTPUT_FONTSIZE/g" \
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
  -e "s/\${NODE_FONTSIZE}/$NODE_FONTSIZE/g" \
  -e "s/\${CLUSTER_FONTSIZE}/$CLUSTER_FONTSIZE/g" \
  -e "s/\${CLUSTER_FONTNAME}/$CLUSTER_FONTNAME/g" \
  -e "s/\${SUBCLUSTER_FONTSIZE}/$SUBCLUSTER_FONTSIZE/g" \
  -e "s/\${SUBCLUSTER_FONTNAME}/$SUBCLUSTER_FONTNAME/g" \
  pamphlet_pipeline.dot.template > "$OUTFILE_DOT"

dot -T${TYPE} "$OUTFILE_DOT" -o "$OUTFILE_TYPE"

# cat "$OUTFILE_DOT"

echo "Dot generated at $OUTFILE_DOT"
echo "$TYPE graph generated at $OUTFILE_TYPE"
