#!/bin/bash
TYPE=svg
OUTFILE="${1:-pamphlet_pipeline_output.$TYPE}"


# https://graphviz.org/doc/info/colors.html#brewer
export COLORSCHEME="gnbu3"

# Set hierarchical colours (1 = lightest/top-level, 2 = subcluster, 3 = nodes)
export PAGE_BG="#DDDDE0"
export CLUSTER_BG="1"
export SUBCLUSTER_BG="2"
export NODE_BG="3"

export CLUSTER_FG="#333333"
export SUBCLUSTER_FG="#333333"
export NODE_FG="#333333"

export FONT_NODE="14"
export FONT_H1="48"
export FONT_H2="32"
export FONT_H3="24"

# Read template and substitute variables
sed \
  -e "s/\${PAGE_BG}/$PAGE_BG/g" \
  -e "s/\${CLUSTER_BG}/$CLUSTER_BG/g" \
  -e "s/\${SUBCLUSTER_BG}/$SUBCLUSTER_BG/g" \
  -e "s/\${NODE_BG}/$NODE_BG/g" \
  -e "s/\${CLUSTER_FG}/$CLUSTER_FG/g" \
  -e "s/\${SUBCLUSTER_FG}/$SUBCLUSTER_FG/g" \
  -e "s/\${NODE_FG}/$NODE_FG/g" \
  -e "s/\${COLORSCHEME}/$COLORSCHEME/g" \
  -e "s/\${FONT_NODE}/$FONT_NODE/g" \
  -e "s/\${FONT_H1}/$FONT_H1/g" \
  -e "s/\${FONT_H2}/$FONT_H2/g" \
  -e "s/\${FONT_H3}/$FONT_H3/g" \
  pamphlet_pipeline.dot.template > /tmp/pamphlet_pipeline.dot

dot -T${TYPE} /tmp/pamphlet_pipeline.dot -o "$OUTFILE"

# cat /tmp/pamphlet_pipeline.dot

echo "Graph generated -> $OUTFILE"
