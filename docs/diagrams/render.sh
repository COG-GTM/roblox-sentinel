#!/usr/bin/env bash
# Render the Graphviz architecture diagram to PNG and SVG.
# Requires Graphviz (e.g. `sudo apt-get install -y graphviz fonts-dejavu`).
set -euo pipefail
cd "$(dirname "$0")"
dot -Tpng -Gdpi=110 architecture.dot -o architecture.png
dot -Tsvg architecture.dot -o architecture.svg
echo "Rendered architecture.png and architecture.svg"
