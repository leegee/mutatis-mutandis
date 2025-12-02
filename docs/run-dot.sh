#!/bin/bash
TYPE=svg
OUTFILE="${1:-pamphlet_pipeline.$TYPE}"


# https://graphviz.org/doc/info/colors.html#brewer
export COLORSCHEME="gnbu4"

# Set hierarchical colours (1 = lightest/top-level, 2 = subcluster, 3 = nodes)
export CLUSTER_BG="1"
export SUBCLUSTER_BG="2"
export NODE_BG="3"

export CLUSTER_FG="#333333"
export SUBCLUSTER_FG="#333333"
export NODE_FG="#333333"

# Read template and substitute variables
sed \
  -e "s/\${CLUSTER_BG}/$CLUSTER_BG/g" \
  -e "s/\${SUBCLUSTER_BG}/$SUBCLUSTER_BG/g" \
  -e "s/\${NODE_BG}/$NODE_BG/g" \
  -e "s/\${CLUSTER_FG}/$CLUSTER_FG/g" \
  -e "s/\${SUBCLUSTER_FG}/$SUBCLUSTER_FG/g" \
  -e "s/\${NODE_FG}/$NODE_FG/g" \
  -e "s/\${COLORSCHEME}/$COLORSCHEME/g" \
  pamphlet_pipeline.dot.template > /tmp/pamphlet_pipeline.dot

dot -Gengine=neato -T${TYPE} /tmp/pamphlet_pipeline.dot -o "$OUTFILE"

# cat /tmp/pamphlet_pipeline.dot

echo "Graph generated -> $OUTFILE"
