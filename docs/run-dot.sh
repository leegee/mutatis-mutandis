#!/bin/bash
TYPE=svg
OUTFILE="${1:-pamphlet_pipeline.$TYPE}"


# https://graphviz.org/doc/info/colors.html#brewer
export COLORSCHEME="gnbu3"

# Set hierarchical colours (1 = lightest/top-level, 2 = subcluster, 3 = nodes)
export CLUSTER_FILL="1"
export SUBCLUSTER_FILL="2"
export NODE_FILL="3"

export CLUSTER_FONT="#333333"
export SUBCLLUSTER_FONT="#333333"
export NODE_FONT="#333333"

# Read template and substitute variables
sed \
  -e "s/\${CLUSTER_FILL}/$CLUSTER_FILL/g" \
  -e "s/\${SUBCLUSTER_FILL}/$SUBCLUSTER_FILL/g" \
  -e "s/\${NODE_FILL}/$NODE_FILL/g" \
  -e "s/\${CLUSTER_FILL}/$CLUSTER_FONT/g" \
  -e "s/\${SUBCLUSTER_FILL}/$SUBCLUSTER_FONT/g" \
  -e "s/\${NODE_FONT}/$NODE_FONT/g" \
  -e "s/\${COLORSCHEME}/$COLORSCHEME/g" \
  pamphlet_pipeline.dot.template > /tmp/pamphlet_pipeline.dot

dot -T${TYPE} /tmp/pamphlet_pipeline.dot -o "$OUTFILE"

cat /tmp/pamphlet_pipeline.dot

echo "Graph generated -> $OUTFILE"
