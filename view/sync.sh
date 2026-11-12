#!/bin/sh
# Pull rollouts and run arguments from fukuyama, then serve the viewer.
cd "$(dirname "$0")/.." || exit 1
rsync -avz --prune-empty-dirs --include='*/' --include='rollouts/*.jsonl' --include='args.json' --exclude='*' \
    fukuyama:projects/cot_illegibility/runs/ runs/
echo "Open http://localhost:8010/view/"
python3 -m http.server 8010
